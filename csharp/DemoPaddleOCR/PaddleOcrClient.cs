using OpenCvSharp;
using Sdcb.PaddleInference;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models.Local;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;

namespace Matsuwa
{
    public class PaddleOcrClient
    {
        private readonly PaddleOcrAll _ocrAll;

        public PaddleOcrClient()
        {
            _ocrAll = new PaddleOcrAll(LocalFullModels.EnglishV4, PaddleDevice.Mkldnn())
            {
                AllowRotateDetection = true,
                Enable180Classification = false,
            };
        }

        public Task<PaddleOcrResult> RecognizeTextAsync(SoftwareBitmap bitmap)
        {
            return Task.Run(() =>
            {
                using Mat src = SoftwareBitmapToMat(bitmap);
                return _ocrAll.Run(src);
            });
        }

        public static void OcrResultToBoxesAndTexts(
            PaddleOcrResult result,
            out List<Rectangle> boxes,
            out string[] texts)
        {
            boxes = [];
            var textList = new List<string>();

            foreach (PaddleOcrResultRegion region in result.Regions)
            {
                RotatedRect rect = region.Rect;
                var vertices = rect.Points();

                float minX = float.MaxValue, minY = float.MaxValue;
                float maxX = float.MinValue, maxY = float.MinValue;
                foreach (var pt in vertices)
                {
                    if (pt.X < minX) minX = pt.X;
                    if (pt.Y < minY) minY = pt.Y;
                    if (pt.X > maxX) maxX = pt.X;
                    if (pt.Y > maxY) maxY = pt.Y;
                }

                boxes.Add(new Rectangle((int)minX, (int)minY, (int)(maxX - minX), (int)(maxY - minY)));
                textList.Add(region.Text);
            }

            texts = [.. textList];
        }

        private static Mat SoftwareBitmapToMat(SoftwareBitmap bitmap)
        {
            // Convert to Bgra8 if needed
            SoftwareBitmap bgraBitmap = bitmap.BitmapPixelFormat == BitmapPixelFormat.Bgra8
                ? bitmap
                : SoftwareBitmap.Convert(bitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);

            int width = bgraBitmap.PixelWidth;
            int height = bgraBitmap.PixelHeight;
            byte[] pixelData = new byte[width * height * 4];

            bgraBitmap.CopyToBuffer(WindowsRuntimeBufferExtensions.AsBuffer(pixelData));

            // Pin the byte array and create a BGR Mat via pointer to avoid CV_8UC4 SetArray issues
            GCHandle handle = GCHandle.Alloc(pixelData, GCHandleType.Pinned);
            try
            {
                using Mat bgra = Mat.FromPixelData(height, width, MatType.CV_8UC4, handle.AddrOfPinnedObject());
                Mat bgr = new Mat();
                Cv2.CvtColor(bgra, bgr, ColorConversionCodes.BGRA2BGR);
                return bgr;
            }
            finally
            {
                handle.Free();
                if (!ReferenceEquals(bgraBitmap, bitmap))
                    bgraBitmap.Dispose();
            }
        }
    }
}
