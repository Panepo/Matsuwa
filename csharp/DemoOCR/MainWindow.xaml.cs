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

        private async void BtnLoadImage_Click(object sender, RoutedEventArgs e)
        {
            var picker = new FileOpenPicker();
            // Associate the picker with the window handle
            var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
            WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

            picker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
            picker.FileTypeFilter.Add(".png");
            picker.FileTypeFilter.Add(".jpg");
            picker.FileTypeFilter.Add(".jpeg");
            picker.FileTypeFilter.Add(".bmp");

            StorageFile file = await picker.PickSingleFileAsync();
            if (file == null) return;

            TxtStatus.Text = $"Loading: {file.Name} …";
            BtnLoadImage.IsEnabled = false;

            try
            {
                using IRandomAccessStream stream = await file.OpenAsync(FileAccessMode.Read);
                SoftwareBitmap convertedImage = await LoadImageAsync(stream);
                if (convertedImage == null)
                {
                    TxtStatus.Text = "Failed to load image.";
                    return;
                }

                await SetImageAsync(SoftwareBitmap.Copy(convertedImage));
                TxtStatus.Text = $"Running OCR on {file.Name} …";
                await RecognizeAndDisplayAsync(convertedImage);
                TxtStatus.Text = $"Done — {file.Name}";
            }
            catch (Exception ex)
            {
                TxtStatus.Text = $"Error: {ex.Message}";
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

            // Apply vehicle info filter
            VehicleInfo vehicleInfo = VehicleInfoFilter.Extract(sb.ToString());
            if (!vehicleInfo.IsEmpty)
            {
                TxtVehicleInfo.Text = vehicleInfo.ToString();
                TxtVehicleInfoHeader.Visibility = Microsoft.UI.Xaml.Visibility.Visible;
                VehicleInfoBorder.Visibility = Microsoft.UI.Xaml.Visibility.Visible;

                string ttsText = BuildVehicleSpeechText(vehicleInfo);
                tts = new WinUITTS("en-US");
                await tts.SynthesisToSpeakerAsync(ttsText, MediaPlayer);
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
                var sb = new StringBuilder("Vehicle detected.");
                if (!string.IsNullOrEmpty(info.Year))  sb.Append($" Year: {info.Year},");
                if (!string.IsNullOrEmpty(info.Make))  sb.Append($" Make: {info.Make},");
                if (!string.IsNullOrEmpty(info.Model)) sb.Append($" Model: {info.Model},");
                if (!string.IsNullOrEmpty(info.Style)) sb.Append($" Style: {info.Style},");
                if (!string.IsNullOrEmpty(info.Color)) sb.Append($" Color: {info.Color}.");
                return sb.ToString();
            }
        }
    }
