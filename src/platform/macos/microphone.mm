/**
 * @file src/platform/macos/microphone.mm
 * @brief Definitions for microphone capture on macOS.
 */
#import <AudioToolbox/AudioToolbox.h>
#import <CoreAudio/CoreAudio.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

// local includes
#include "src/config.h"
#include "src/logging.h"
#include "src/platform/common.h"
#include "src/platform/macos/av_audio.h"
#include "src/utility.h"

namespace platf {
  using namespace std::literals;

  /**
   * @brief macOS microphone capture device and audio format state.
   */
  struct av_mic_t: public mic_t {
    AVAudio *av_audio_capture {};  ///< AV audio capture.

    ~av_mic_t() override {
      [av_audio_capture release];
    }

    /**
     * @brief Deliver a captured audio sample to Sunshine's audio pipeline.
     *
     * @param sample_in Sample in.
     * @return Capture status reported to the streaming pipeline.
     */
    capture_e sample(std::vector<float> &sample_in) override {
      const uint32_t neededBytes = static_cast<uint32_t>(sample_in.size() * sizeof(float));
      uint8_t *dst = reinterpret_cast<uint8_t *>(sample_in.data());

      uint32_t remaining = neededBytes;

      while (remaining > 0) {
        uint32_t avail = 0;
        void *tail = TPCircularBufferTail(&av_audio_capture->audioSampleBuffer, &avail);

        if (avail == 0) {
          // Using 5 second timeout to prevent indefinite hanging
          dispatch_time_t timeout = dispatch_time(DISPATCH_TIME_NOW, 5LL * NSEC_PER_SEC);
          if (dispatch_semaphore_wait(av_audio_capture->audioSemaphore, timeout) != 0) {
            BOOST_LOG(warning) << "Audio sample timeout - no audio data received within 5 seconds"sv;

            // Fill with silence and return to prevent hanging
            std::fill(sample_in.begin(), sample_in.end(), 0.0f);
            return capture_e::timeout;
          }
          continue;
        }

        const uint32_t toCopy = (avail < remaining) ? avail : remaining;
        std::memcpy(dst, tail, toCopy);

        TPCircularBufferConsume(&av_audio_capture->audioSampleBuffer, toCopy);

        dst += toCopy;
        remaining -= toCopy;
      }

      return capture_e::ok;
    }
  };

  /**
   * @brief Plays the client's microphone into a virtual audio device, so that host
   *        applications can record it from that device's input side.
   *
   * write_pcm() is the only producer and the device's IO thread the only consumer,
   * which is the one-reader-one-writer case TPCircularBuffer is safe for without a lock.
   */
  struct mic_redirect_t {
    AudioComponentInstance unit {};
    AudioDeviceID device_id { kAudioObjectUnknown };
    Float64 sample_rate { 48000.0 };
    TPCircularBuffer ring {};
    UInt32 channels { 2 };
    std::atomic<bool> stopping { false };
    std::atomic<int> active_renders { 0 };

    ~mic_redirect_t() {
      stopping.store(true, std::memory_order_release);
      if (unit) {
        AURenderCallbackStruct empty_cb { nullptr, nullptr };
        AudioUnitSetProperty(unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &empty_cb, sizeof(empty_cb));
        AudioOutputUnitStop(unit);
        AudioUnitUninitialize(unit);
        AudioComponentInstanceDispose(unit);
        unit = nullptr;
      }
      while (active_renders.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();
      }
      TPCircularBufferCleanup(&ring);
    }

    bool
    is_device_alive() const {
      if (device_id == kAudioObjectUnknown) {
        return false;
      }
      AudioObjectPropertyAddress address {
        kAudioDevicePropertyDeviceIsAlive,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
      };
      UInt32 alive = 0;
      UInt32 size = sizeof(alive);
      if (AudioObjectGetPropertyData(device_id, &address, 0, nullptr, &size, &alive) != noErr || alive == 0) {
        return false;
      }
      return true;
    }

    static std::string
    string_property(AudioDeviceID device, AudioObjectPropertySelector selector) {
      AudioObjectPropertyAddress address { selector, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
      CFStringRef value = nullptr;
      UInt32 size = sizeof(value);
      if (AudioObjectGetPropertyData(device, &address, 0, nullptr, &size, &value) != noErr || !value) {
        return {};
      }
      CFIndex utf8_max = CFStringGetMaximumSizeForEncoding(CFStringGetLength(value), kCFStringEncodingUTF8) + 1;
      std::string text(utf8_max, '\0');
      if (CFStringGetCString(value, text.data(), utf8_max, kCFStringEncodingUTF8)) {
        text.resize(std::strlen(text.c_str()));
      }
      else {
        text.clear();
      }
      CFRelease(value);
      return text;
    }

    /// The output device named by `virtual_sink` (name or UID); without that setting, the first virtual one.
    static AudioDeviceID
    find_device(const std::string &wanted) {
      AudioObjectPropertyAddress address { kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
      UInt32 size = 0;
      if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &address, 0, nullptr, &size) != noErr) {
        return kAudioObjectUnknown;
      }
      std::vector<AudioDeviceID> devices(size / sizeof(AudioDeviceID));
      if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &address, 0, nullptr, &size, devices.data()) != noErr) {
        return kAudioObjectUnknown;
      }

      if (!wanted.empty()) {
        // Pass 1: exact match on Device UID
        for (auto device : devices) {
          AudioObjectPropertyAddress streams { kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain };
          UInt32 streams_size = 0;
          if (AudioObjectGetPropertyDataSize(device, &streams, 0, nullptr, &streams_size) != noErr || streams_size == 0) {
            continue;
          }
          if (string_property(device, kAudioDevicePropertyDeviceUID) == wanted) {
            return device;
          }
        }
        // Pass 2: match on Device Name
        for (auto device : devices) {
          AudioObjectPropertyAddress streams { kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain };
          UInt32 streams_size = 0;
          if (AudioObjectGetPropertyDataSize(device, &streams, 0, nullptr, &streams_size) != noErr || streams_size == 0) {
            continue;
          }
          if (string_property(device, kAudioObjectPropertyName) == wanted) {
            return device;
          }
        }
        return kAudioObjectUnknown;
      }

      // Default: first device with virtual transport type
      for (auto device : devices) {
        AudioObjectPropertyAddress streams { kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain };
        UInt32 streams_size = 0;
        if (AudioObjectGetPropertyDataSize(device, &streams, 0, nullptr, &streams_size) != noErr || streams_size == 0) {
          continue;  // no output side
        }

        AudioObjectPropertyAddress transport_address { kAudioDevicePropertyTransportType, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
        UInt32 transport = 0;
        UInt32 transport_size = sizeof(transport);
        if (AudioObjectGetPropertyData(device, &transport_address, 0, nullptr, &transport_size, &transport) == noErr &&
            transport == kAudioDeviceTransportTypeVirtual) {
          return device;
        }
      }
      return kAudioObjectUnknown;
    }

    static OSStatus
    render(void *self_ptr, AudioUnitRenderActionFlags *, const AudioTimeStamp *, UInt32, UInt32, AudioBufferList *data) {
      auto *self = static_cast<mic_redirect_t *>(self_ptr);
      if (!self || self->stopping.load(std::memory_order_acquire)) {
        if (data && data->mNumberBuffers > 0) {
          std::memset(data->mBuffers[0].mData, 0, data->mBuffers[0].mDataByteSize);
        }
        return noErr;
      }

      self->active_renders.fetch_add(1, std::memory_order_acquire);
      auto guard = util::fail_guard([&]() {
        self->active_renders.fetch_sub(1, std::memory_order_release);
      });

      if (self->stopping.load(std::memory_order_acquire)) {
        if (data && data->mNumberBuffers > 0) {
          std::memset(data->mBuffers[0].mData, 0, data->mBuffers[0].mDataByteSize);
        }
        return noErr;
      }

      auto &out = data->mBuffers[0];

      uint32_t available = 0;
      void *tail = TPCircularBufferTail(&self->ring, &available);
      const auto copied = std::min<uint32_t>(available, out.mDataByteSize);
      if (copied) {
        std::memcpy(out.mData, tail, copied);
        TPCircularBufferConsume(&self->ring, copied);
      }
      std::memset(static_cast<char *>(out.mData) + copied, 0, out.mDataByteSize - copied);  // silence when the client sends nothing
      return noErr;
    }

    static std::unique_ptr<mic_redirect_t>
    open(const std::string &wanted) {
      auto device = find_device(wanted);
      if (device == kAudioObjectUnknown) {
        BOOST_LOG(warning) << "Microphone redirect: no "sv << (wanted.empty() ? "virtual audio output device"s : "output device named ["s + wanted + ']')
                           << " found. Install a virtual audio driver, or set 'virtual_sink'."sv;
        return nullptr;
      }

      AudioComponentDescription description { kAudioUnitType_Output, kAudioUnitSubType_HALOutput, kAudioUnitManufacturer_Apple };
      auto self = std::make_unique<mic_redirect_t>();
      if (AudioComponentInstanceNew(AudioComponentFindNext(nullptr, &description), &self->unit) != noErr) {
        self->unit = nullptr;
        return nullptr;
      }

      AudioStreamBasicDescription format {};
      UInt32 size = sizeof(format);
      bool ok = AudioUnitSetProperty(self->unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &device, sizeof(device)) == noErr;
      Float64 device_sample_rate = 48000.0;
      if (ok && AudioUnitGetProperty(self->unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 0, &format, &size) == noErr) {
        if (format.mChannelsPerFrame) {
          self->channels = format.mChannelsPerFrame;
        }
        if (format.mSampleRate > 0) {
          device_sample_rate = format.mSampleRate;
        }
      }

      // Buffer capacity scales with channel count to comfortably hold at least 500ms of audio
      const auto buffer_capacity = std::max<std::size_t>(128 * 1024, self->channels * sizeof(float) * static_cast<std::size_t>(device_sample_rate / 2.0));
      if (!TPCircularBufferInit(&self->ring, buffer_capacity)) {
        BOOST_LOG(error) << "Microphone redirect: failed to allocate circular buffer for "sv << self->channels << " channels"sv;
        return nullptr;
      }

      self->device_id = device;
      self->sample_rate = device_sample_rate;

      format = {};
      format.mSampleRate = device_sample_rate;
      format.mFormatID = kAudioFormatLinearPCM;
      format.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
      format.mChannelsPerFrame = self->channels;
      format.mBitsPerChannel = 32;
      format.mFramesPerPacket = 1;
      format.mBytesPerFrame = format.mBytesPerPacket = self->channels * sizeof(float);
      AURenderCallbackStruct callback { render, self.get() };
      ok = ok && AudioUnitSetProperty(self->unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &format, sizeof(format)) == noErr &&
           AudioUnitSetProperty(self->unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &callback, sizeof(callback)) == noErr &&
           AudioUnitInitialize(self->unit) == noErr &&
           AudioOutputUnitStart(self->unit) == noErr;
      if (!ok) {
        BOOST_LOG(error) << "Microphone redirect: couldn't open ["sv << string_property(device, kAudioObjectPropertyName) << ']';
        return nullptr;
      }

      BOOST_LOG(info) << "Microphone redirect: client microphone plays into ["sv << string_property(device, kAudioObjectPropertyName) << ']';
      return self;
    }

    /// @param samples Mono 48 kHz samples, copied to every channel of the device.
    int
    write_pcm(const std::int16_t *samples, std::size_t frame_count) {
      if (!unit) {
        return -1;
      }
      if (!is_device_alive()) {
        return -2;  // Device invalid or removed: signal stream.cpp to reopen
      }
      std::size_t out_frames = frame_count;
      if (sample_rate != 48000.0 && sample_rate > 0) {
        out_frames = static_cast<std::size_t>(std::round(frame_count * (sample_rate / 48000.0)));
      }
      std::vector<float> out(out_frames * channels);
      const double step = static_cast<double>(frame_count) / static_cast<double>(out_frames);
      for (std::size_t f = 0; f < out_frames; ++f) {
        double src_idx = f * step;
        std::size_t idx0 = static_cast<std::size_t>(src_idx);
        std::size_t idx1 = std::min(idx0 + 1, frame_count - 1);
        double frac = src_idx - idx0;
        float sample = (samples[idx0] * (1.0 - frac) + samples[idx1] * frac) / 32768.0f;
        for (UInt32 c = 0; c < channels; ++c) {
          out[f * channels + c] = sample;
        }
      }
      if (!TPCircularBufferProduceBytes(&ring, out.data(), out.size() * sizeof(float))) {
        return 0;  // Ring buffer full: drop frame gracefully, return 0 consumed
      }
      return static_cast<int>(frame_count * sizeof(std::int16_t));
    }
  };

  /**
   * @brief macOS audio control state used to create microphone streams.
   */
  struct macos_audio_control_t: public audio_control_t {
    AVCaptureDevice *audio_capture_device {};  ///< Audio capture device.
    std::unique_ptr<mic_redirect_t> mic_redirect;

  public:
    /**
     * @brief Update the sink value on the backend.
     *
     * @param sink Audio sink name to route or capture.
     * @return Status from updating sink.
     */
    int set_sink(const std::string &sink) override {
      BOOST_LOG(warning) << "audio_control_t::set_sink() unimplemented: "sv << sink;
      return 0;
    }

    /**
     * @brief Create a microphone capture stream for the requested layout.
     *
     * @param mapping Opus channel mapping table for the requested layout.
     * @param channels Number of audio channels in the stream.
     * @param sample_rate Audio sample rate in hertz.
     * @param frame_size Number of samples captured per audio frame.
     * @param continuous_audio Continuous audio.
     * @param host_audio_enabled Whether host playback should remain enabled during capture.
     * @return Microphone capture object for the requested audio layout.
     */
    std::unique_ptr<mic_t> microphone(const std::uint8_t *mapping, int channels, std::uint32_t sample_rate, std::uint32_t frame_size, bool continuous_audio, bool host_audio_enabled) override {
      auto mic = std::make_unique<av_mic_t>();
      mic->av_audio_capture = [[AVAudio alloc] init];

      // Set the host audio enabled flag from the stream configuration
      mic->av_audio_capture.hostAudioEnabled = host_audio_enabled ? YES : NO;
      BOOST_LOG(debug) << "Set hostAudioEnabled to: "sv << (host_audio_enabled ? "YES" : "NO");

      if (config::audio.sink.empty()) {
        // Use macOS system-wide audio tap
        BOOST_LOG(info) << "Using macOS system audio tap for capture."sv;
        BOOST_LOG(info) << "Sample rate: "sv << sample_rate << ", Frame size: "sv << frame_size << ", Channels: "sv << channels;

        if ([mic->av_audio_capture setupSystemTap:sample_rate frameSize:frame_size channels:channels]) {
          BOOST_LOG(error) << "Failed to setup system audio tap."sv;
          return nullptr;
        }
      } else {
        // Use specified macOS audio sink
        const char *audio_sink = config::audio.sink.c_str();
        BOOST_LOG(info) << "Using configured audio sink "sv << audio_sink << " for capture."sv;

        if (!request_microphone_permission()) {
          return nullptr;
        }

        if ((audio_capture_device = [AVAudio findMicrophone:[NSString stringWithUTF8String:audio_sink]]) == nullptr) {
          BOOST_LOG(error) << "opening microphone '"sv << audio_sink << "' failed. Please set a valid input source in the Sunshine config."sv;
          BOOST_LOG(error) << "Available inputs:"sv;

          for (NSString *name in [AVAudio microphoneNames]) {
            BOOST_LOG(error) << "\t"sv << [name UTF8String];
          }

          return nullptr;
        }

        if ([mic->av_audio_capture setupMicrophone:audio_capture_device sampleRate:sample_rate frameSize:frame_size channels:channels]) {
          BOOST_LOG(error) << "Failed to setup microphone."sv;
          return nullptr;
        }
      }

      return mic;
    }

    bool is_sink_available(const std::string &sink) override {
      BOOST_LOG(warning) << "audio_control_t::is_sink_available() unimplemented: "sv << sink;
      return true;
    }

    /**
     * @brief Query host and virtual sink names available to Sunshine.
     *
     * @return Host and virtual sink names when the backend can report them.
     */
    std::optional<sink_t> sink_info() override {
      sink_t sink;

      return sink;
    }

    int write_mic_pcm(const std::int16_t *samples, std::size_t frame_count) override {
      return mic_redirect ? mic_redirect->write_pcm(samples, frame_count) : -1;
    }

    int init_mic_redirect_device() override {
      if (!mic_redirect) {
        mic_redirect = mic_redirect_t::open(config::audio.virtual_sink);
      }
      return mic_redirect ? 0 : -1;
    }

    void release_mic_redirect_device() override {
      mic_redirect.reset();
    }
  };

  std::unique_ptr<audio_control_t> audio_control() {
    return std::make_unique<macos_audio_control_t>();
  }
}  // namespace platf
