using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Diagnostics;
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;

namespace Matsuwa
{
    public partial class AzureTTSWinUI
    {
        private static readonly string EmbeddedSpeechSynthesisVoicePath = "./TTSmodels";

        public enum AzureTtsStatus
        {
            NOT_INITIAL,
            READY,
            ERROR_INITIAL,

            SYNTHESIZING,
            SYNTHESIZED,
            NOT_SYNTHESIZED,
            CANCELED,
            SESSION_START,
            SESSION_STOP
        }

        public enum AzureTtsVoiceName
        {
            // danish voices
            DA_DK_CHRISTEL,
            DA_DK_JEPPE,

            // german voices
            DE_DE_CONRAD,
            DE_DE_KATJA,

            // english voices
            EN_US_ARIA,
            EN_US_JENNY,
            EN_US_GUY,

            // french voices
            FR_FR_DENIS,
            FR_FR_HENRI,

            // italian voices
            IT_IT_DIEGO,
            IT_IT_ISABELLA,

            // japanese voices
            JA_JP_KEITA,
            JA_JP_NANAMI,

            // korean voices
            KO_KR_INJOON,
            KO_KR_SUNHI,

            // dutch voices
            NI_NL_FENNA,
            NI_NL_MAARTEN,

            // portuguese voices
            PT_PT_DUARTE,
            PT_PT_RAQUEL,

            // russian voices
            RU_RU_DMITRY,
            RU_RU_SVETLANA,

            // spanish voice
            ES_ES_ALVARO,
            ES_ES_ELVIRA,

            // swedish voices
            SV_SE_MATTIAS,
            SV_SE_SOFIA,

            // chinese voices
            ZH_CN_XIAOXIAO,
            ZH_CN_YUNXI,

            // chinese (taiwan) voices
            ZH_TW_HSAIOCHEN,
            ZH_TW_YUNJHE,
        }

        private static string GetAzureTtsVoiceName(AzureTtsVoiceName voice)
        {
            switch (voice)
            {
                // Danish
                case AzureTtsVoiceName.DA_DK_CHRISTEL:
                    return "da-DK-ChristelNeural";
                case AzureTtsVoiceName.DA_DK_JEPPE:
                    return "da-DK-JeppeNeural";

                // German
                case AzureTtsVoiceName.DE_DE_CONRAD:
                    return "de-DE-ConradNeural";
                case AzureTtsVoiceName.DE_DE_KATJA:
                    return "de-DE-KatjaNeural";

                // English
                case AzureTtsVoiceName.EN_US_ARIA:
                    return "en-US-AriaNeural";
                case AzureTtsVoiceName.EN_US_JENNY:
                    return "en-US-JennyNeural";
                case AzureTtsVoiceName.EN_US_GUY:
                    return "en-US-GuyNeural";

                // French
                case AzureTtsVoiceName.FR_FR_DENIS:
                    return "fr-FR-DenisNeural";
                case AzureTtsVoiceName.FR_FR_HENRI:
                    return "fr-FR-HenriNeural";

                // Italian
                case AzureTtsVoiceName.IT_IT_DIEGO:
                    return "it-IT-DiegoNeural";
                case AzureTtsVoiceName.IT_IT_ISABELLA:
                    return "it-IT-IsabellaNeural";

                // Japanese
                case AzureTtsVoiceName.JA_JP_KEITA:
                    return "ja-JP-KeitaNeural";
                case AzureTtsVoiceName.JA_JP_NANAMI:
                    return "ja-JP-NanamiNeural";

                // Korean
                case AzureTtsVoiceName.KO_KR_INJOON:
                    return "ko-KR-InJoonNeural";
                case AzureTtsVoiceName.KO_KR_SUNHI:
                    return "ko-KR-SunHiNeural";

                // Dutch
                case AzureTtsVoiceName.NI_NL_FENNA:
                    return "nl-NL-FennaNeural";
                case AzureTtsVoiceName.NI_NL_MAARTEN:
                    return "nl-NL-MaartenNeural";

                // Portuguese
                case AzureTtsVoiceName.PT_PT_DUARTE:
                    return "pt-PT-DuarteNeural";
                case AzureTtsVoiceName.PT_PT_RAQUEL:
                    return "pt-PT-RaquelNeural";

                // Russian
                case AzureTtsVoiceName.RU_RU_DMITRY:
                    return "ru-RU-DmitryNeural";
                case AzureTtsVoiceName.RU_RU_SVETLANA:
                    return "ru-RU-SvetlanaNeural";

                // Swedish
                case AzureTtsVoiceName.SV_SE_MATTIAS:
                    return "sv-SE-MattiasNeural";
                case AzureTtsVoiceName.SV_SE_SOFIA:
                    return "sv-SE-SofiaNeural";

                // Chinese (Mainland)
                case AzureTtsVoiceName.ZH_CN_XIAOXIAO:
                    return "zh-CN-XiaoxiaoNeural";
                case AzureTtsVoiceName.ZH_CN_YUNXI:
                    return "zh-CN-YunxiNeural";

                // Chinese (Taiwan)
                case AzureTtsVoiceName.ZH_TW_HSAIOCHEN:
                    return "zh-TW-HsiaoChenNeural";
                case AzureTtsVoiceName.ZH_TW_YUNJHE:
                    return "zh-TW-YunJheNeural";

                default:
                    throw new ArgumentOutOfRangeException(nameof(voice), voice, null);
            }
        }

        public struct AzureTtsOutput
        {
            public AzureTtsStatus status;
            public string message;
        }

        public class ProcessEventArgs : EventArgs
        {
            public AzureTtsOutput Output { get; set; }
        }

        private readonly AzureTtsStatus _status = AzureTtsStatus.NOT_INITIAL;
        public AzureTtsStatus Status
        {
            get { return _status; }
        }

        private readonly SpeechSynthesizer _synthesizer;
        private AzureTtsOutput _output;
        private float _volume = 1.0f;

        public delegate void ProcessEventHandler(object mObjct, ProcessEventArgs mArgs);
        public event ProcessEventHandler? OnProcessed;

        public AzureTTSWinUI(IntPtr EmbeddedSpeechSynthesisVoiceKey, AzureTtsVoiceName voice, string deviceID)
        {
            try
            {
                string EmbeddedSpeechSynthesisVoiceName = GetAzureTtsVoiceName(voice);
                string? entryAssemblyLocation = Assembly.GetEntryAssembly()?.Location;
                if (string.IsNullOrEmpty(entryAssemblyLocation))
                    throw new InvalidOperationException("Entry assembly location is null. Cannot determine current path.");
                string CurrentPath = Path.GetDirectoryName(entryAssemblyLocation)!;
                EmbeddedSpeechConfig embConfig = EmbeddedSpeechConfig.FromPath(Path.Combine(CurrentPath, EmbeddedSpeechSynthesisVoicePath));
                embConfig.SetSpeechSynthesisVoice(EmbeddedSpeechSynthesisVoiceName, Marshal.PtrToStringAuto(EmbeddedSpeechSynthesisVoiceKey));
                if (EmbeddedSpeechSynthesisVoiceName.Contains("Neural"))
                {
                    // Embedded neural voices only support 24kHz and the engine has no ability to resample.
                    embConfig.SetSpeechSynthesisOutputFormat(SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm);
                }

                AudioConfig audioOutput = string.IsNullOrEmpty(deviceID)
                    ? AudioConfig.FromDefaultSpeakerOutput()
                    : AudioConfig.FromSpeakerOutput(deviceID);
                _synthesizer = new SpeechSynthesizer(embConfig, audioOutput);

                _synthesizer.SynthesisStarted += HandleSessionStart;
                _synthesizer.Synthesizing += HandleSynthesizing;
                _synthesizer.SynthesisCompleted += HandleSynthesisCompleted;

                _status = AzureTtsStatus.READY;
            }
            catch (Exception ex)
            {
                _status = AzureTtsStatus.ERROR_INITIAL;
                throw new ApplicationException(ex.Message);
            }
        }

        public async Task ListEmbeddedVoicesAsync()
        {
            using SynthesisVoicesResult result = await _synthesizer.GetVoicesAsync("");

            if (result.Reason == ResultReason.VoicesListRetrieved)
            {
                Debug.WriteLine("Voices found:");
                foreach (var voice in result.Voices)
                {
                    Debug.WriteLine(voice.Name);
                    Debug.WriteLine($" Gender: {voice.Gender}");
                    Debug.WriteLine($" Locale: {voice.Locale}");
                    Debug.WriteLine($" Path:   {voice.VoicePath}");
                }
            }
            else if (result.Reason == ResultReason.Canceled)
            {
                Debug.WriteLine($"CANCELED: ErrorDetails=\"{result.ErrorDetails}\"");
            }
        }

        public void Release()
        {
            _synthesizer.SynthesisStarted -= HandleSessionStart;
            _synthesizer.Synthesizing -= HandleSynthesizing;
            _synthesizer.Dispose();
        }

        public void SynthesisToSpeakerAsync(string text)
        {
            if (_synthesizer == null) throw new ApplicationException("Speech synthesizer is not initialized.");

            ProcessEventArgs eventArgs = new();

            if (_status == AzureTtsStatus.READY)
            {
                // Apply volume using SSML
                // Volume in SSML is specified as a percentage (0-100) or as relative values
                string volumePercent = (_volume * 100).ToString("F0");
                string ssml = $@"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
                    <voice name='{_synthesizer.Properties.GetProperty(PropertyId.SpeechServiceConnection_SynthVoice)}'>
                        <prosody volume='{volumePercent}'>
                            {System.Security.SecurityElement.Escape(text)}
                        </prosody>
                    </voice>
                </speak>";

                Task<SpeechSynthesisResult> task = _synthesizer.SpeakSsmlAsync(ssml);
                task.Wait();
                var result = task.Result;
                {
                    if (result.Reason == ResultReason.SynthesizingAudioCompleted)
                    {
                        _output.message = $"Speech synthesized to speaker for text [{text}]";
                        _output.status = AzureTtsStatus.SYNTHESIZED;
                        eventArgs.Output = _output;
                        OnProcessed?.Invoke(this, eventArgs);
                    }
                    else if (result.Reason == ResultReason.Canceled)
                    {
                        var cancellation = SpeechSynthesisCancellationDetails.FromResult(result);

                        _output.status = AzureTtsStatus.CANCELED;
                        _output.message = $"CANCELED: Reason={cancellation.Reason} ErrorCode={cancellation.ErrorCode} ErrorDetails=\"{cancellation.ErrorDetails}\"";
                        eventArgs.Output = _output;
                        OnProcessed?.Invoke(this, eventArgs);
                    }
                }
            }
            else
            {
                _output.status = AzureTtsStatus.ERROR_INITIAL;
                _output.message = "AzureTTs initial failed";
                eventArgs.Output = _output;
                OnProcessed?.Invoke(this, eventArgs);
            }
        }

        public void StopSynthesis()
        {
            _synthesizer.StopSpeakingAsync();
        }

        /// <summary>
        /// Set the output volume for speech synthesis.
        /// </summary>
        /// <param name="volume">Volume level (0.0 to 1.0, where 1.0 is 100%)</param>
        public void SetVolume(float volume)
        {
            _volume = Math.Clamp(volume, 0.0f, 1.0f);
            System.Diagnostics.Debug.WriteLine($"TTS volume set to: {_volume:F2}");
        }

        /// <summary>
        /// Get the current volume setting.
        /// </summary>
        /// <returns>Volume level (0.0 to 1.0)</returns>
        public float GetVolume()
        {
            return _volume;
        }

        private void HandleSessionStart(object? sender, SpeechSynthesisEventArgs e)
        {
            _output.status = AzureTtsStatus.SESSION_START;
            _output.message = "Synthesis started event.";

            ProcessEventArgs eventArgs = new()
            {
                Output = _output
            };
            OnProcessed?.Invoke(this, eventArgs);
        }

        private void HandleSynthesizing(object? sender, SpeechSynthesisEventArgs e)
        {
            _output.status = AzureTtsStatus.SYNTHESIZING;
            _output.message = $"Synthesizing, received an audio chunk of {e.Result.AudioData.Length} bytes.";

            ProcessEventArgs eventArgs = new() { Output = _output };
            OnProcessed?.Invoke(this, eventArgs);
        }

        private void HandleSynthesisCompleted(object? sender, SpeechSynthesisEventArgs e)
        {
            _output.status = AzureTtsStatus.SYNTHESIZED;
            _output.message = $"Synthesized.";

            ProcessEventArgs eventArgs = new() { Output = _output };
            OnProcessed?.Invoke(this, eventArgs);
        }
    }
}