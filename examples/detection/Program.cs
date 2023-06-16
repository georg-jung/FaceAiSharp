using FaceAiSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using var hc = new HttpClient();
var groupPhoto = await hc.GetByteArrayAsync(
    "https://raw.githubusercontent.com/georg-jung/FaceAiSharp/master/examples/obama_family.jpg");
var img = Image.Load<Rgb24>(groupPhoto);

var det = FaceAiSharpBundleFactory.CreateFaceDetector();
var faces = det.DetectFaces(img);
foreach (var face in faces)
{
    Console.WriteLine($"Found a face with conficence {face.Confidence}: {face.Box}");
}
