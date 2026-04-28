using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
using Windows.Media.SpeechSynthesis;
using Windows.Media.Core;
using Microsoft.UI.Xaml.Controls;


namespace Matsuwa
{
    public class WinUITTS
    {
        private SpeechSynthesizer _synthesizer;
        public WinUITTS(string language)
        {
            _synthesizer = new SpeechSynthesizer();

            var voice = SpeechSynthesizer.AllVoices.FirstOrDefault(v => v.Language.StartsWith(language));
            if (voice != null)
            {
                _synthesizer.Voice = voice;
            }
            else
            {
                voice = SpeechSynthesizer.AllVoices.FirstOrDefault(v => v.Language.StartsWith("en-Us"));
                _synthesizer.Voice = voice;
            }
        }

        public void SetVoice(string language)
        {
            if (_synthesizer == null) throw new ApplicationException("Speech synthesizer is not initialized.");
            var voice = SpeechSynthesizer.AllVoices.FirstOrDefault(v => v.Language.StartsWith(language));
            if (voice != null)
            {
                _synthesizer.Voice = voice;
            }
            else
            {
                throw new ArgumentException($"No voice found for language: {language}");
            }
        }

        public static void GetVoice()
        {
            foreach (var voice in SpeechSynthesizer.AllVoices)
            {
                Debug.WriteLine($"{voice.DisplayName} - {voice.Language}");
            }
        }

        public async Task SynthesisToSpeakerAsync(string text, MediaPlayerElement media)
        {
            if (_synthesizer == null) throw new ApplicationException("Speech synthesizer is not initialized.");
            if (string.IsNullOrWhiteSpace(text)) return;

            SpeechSynthesisStream synthesisStream = await _synthesizer.SynthesizeTextToStreamAsync(text);

            media.DispatcherQueue.TryEnqueue(() =>
            {
                media.Source = MediaSource.CreateFromStream(synthesisStream, synthesisStream.ContentType);
                media.MediaPlayer.Play();
            });
        }
    }
}
