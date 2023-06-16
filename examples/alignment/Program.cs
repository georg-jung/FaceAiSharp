using FaceAiSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using var hc = new HttpClient();
var groupPhoto = await hc.GetByteArrayAsync(
    "https://raw.githubusercontent.com/georg-jung/FaceAiSharp/master/examples/obama_family.jpg");
var img = Image.Load<Rgb24>(groupPhoto);

var det = FaceAiSharpBundleFactory.CreateFaceDetectorWithLandmarks();
var rec = FaceAiSharpBundleFactory.CreateFaceEmbeddingsGenerator();

var face = det.DetectFaces(img).First();
rec.AlignFaceUsingLandmarks(img, face.Landmarks);
img.Save("aligned.jpg");
