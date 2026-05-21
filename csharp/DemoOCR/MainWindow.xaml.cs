using Matsuwa;
using Microsoft.UI;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using Microsoft.UI.Xaml.Shapes;
using System;
using System.Text;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;

namespace DemoOCR
{
    public sealed partial class MainWindow : Window
    {
        private readonly TextRecoClient _textRecoClient = new();
        private WinUITTS tts = null!;

        public MainWindow()
        {
            InitializeComponent();
        }

        private VehicleInfo _accumulatedInfo = new VehicleInfo();

        private async void BtnLoadImage_Click(object sender, RoutedEventArgs e)
        {
            var picker = new FolderPicker();
            var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
            WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

            picker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
            picker.FileTypeFilter.Add("*");

            StorageFolder folder = await picker.PickSingleFolderAsync();
            if (folder == null) return;

            // Reset accumulated info for a new session
            _accumulatedInfo = new VehicleInfo();

            var files = await folder.GetFilesAsync();
            BtnLoadImage.IsEnabled = false;

            try
            {
                foreach (StorageFile file in files)
                {
                    string ext = file.FileType.ToLowerInvariant();
                    if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp")
                        continue;

                    if (_accumulatedInfo.IsComplete) break;

                    TxtStatus.Text = $"Processing: {file.Name} …";

                    try
                    {
                        using IRandomAccessStream stream = await file.OpenAsync(FileAccessMode.Read);
                        SoftwareBitmap convertedImage = await LoadImageAsync(stream);
                        if (convertedImage == null) continue;

                        await SetImageAsync(SoftwareBitmap.Copy(convertedImage));
                        await RecognizeAndDisplayAsync(convertedImage);
                        TxtStatus.Text = _accumulatedInfo.IsComplete
                            ? $"Complete — all info found in {file.Name}"
                            : $"Processed: {file.Name}";
                    }
                    catch (Exception ex)
                    {
                        TxtStatus.Text = $"Error processing {file.Name}: {ex.Message}";
                    }
                }

                if (!_accumulatedInfo.IsComplete)
                    TxtStatus.Text = "Finished — some vehicle info could not be found.";
            }
            finally
            {
                BtnLoadImage.IsEnabled = true;
            }
        }

        private static async Task<SoftwareBitmap> LoadImageAsync(IRandomAccessStream stream)
        {
            var decoder = await BitmapDecoder.CreateAsync(stream);
            SoftwareBitmap inputBitmap = await decoder.GetSoftwareBitmapAsync(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
            if (inputBitmap == null) return null;

            // Ensure Bgra8 + Premultiplied (required by SoftwareBitmapSource)
            return SoftwareBitmap.Convert(inputBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
        }

        private async Task SetImageAsync(SoftwareBitmap convertedImage)
        {
            RectCanvas.Children.Clear();
            TxtResult.Text = string.Empty;

            var bitmapSource = new SoftwareBitmapSource();
            await bitmapSource.SetBitmapAsync(convertedImage);
            ImageSrc.Source = bitmapSource;

            // Size the canvas to match the image
            RectCanvas.Width = convertedImage.PixelWidth;
            RectCanvas.Height = convertedImage.PixelHeight;
        }

        private async Task RecognizeAndDisplayAsync(SoftwareBitmap bitmap)
        {
            var recognizedText = await _textRecoClient.RecognizeTextAsync(bitmap);

            TextRecoClient.RecognizedTextToBoxesAndTexts(recognizedText, out var boxes, out var texts);

            var sb = new StringBuilder();
            for (int i = 0; i < texts.Length; i++)
            {
                // Draw bounding rectangle on canvas
                var rect = new Rectangle
                {
                    Width = boxes[i].Width,
                    Height = boxes[i].Height,
                    Stroke = new SolidColorBrush(Colors.DeepSkyBlue),
                    StrokeThickness = 2,
                    Fill = new SolidColorBrush(Windows.UI.Color.FromArgb(30, 0, 191, 255))
                };
                Microsoft.UI.Xaml.Controls.Canvas.SetLeft(rect, boxes[i].X);
                Microsoft.UI.Xaml.Controls.Canvas.SetTop(rect, boxes[i].Y);
                RectCanvas.Children.Add(rect);

                sb.AppendLine(texts[i]);
            }

            TxtResult.Text = sb.ToString();

            // Apply vehicle info filter and merge into accumulated info
            VehicleInfo vehicleInfo = VehicleInfoFilter.Extract(sb.ToString());
            _accumulatedInfo.MergeFrom(vehicleInfo);

            if (!_accumulatedInfo.IsEmpty)
            {
                TxtVehicleInfo.Text = _accumulatedInfo.ToString();
                TxtVehicleInfoHeader.Visibility = Microsoft.UI.Xaml.Visibility.Visible;
                VehicleInfoBorder.Visibility = Microsoft.UI.Xaml.Visibility.Visible;

                if (_accumulatedInfo.IsComplete)
                {
                    string ttsText = BuildVehicleSpeechText(_accumulatedInfo);
                    tts = new WinUITTS("en-US");
                    await tts.SynthesisToSpeakerAsync(ttsText, MediaPlayer);
                }
            }
            else
            {
                TxtVehicleInfo.Text = string.Empty;
                TxtVehicleInfoHeader.Visibility = Microsoft.UI.Xaml.Visibility.Collapsed;
                VehicleInfoBorder.Visibility = Microsoft.UI.Xaml.Visibility.Collapsed;
            }
        }
            private static string BuildVehicleSpeechText(VehicleInfo info)
            {
                return $"{info.Color}, {info.Year}, {info.Make}, EXPIRES {info.Expire}, Insurance {info.Insurance}";
            }
        }
    }
