using FaceAiSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using var hc = new HttpClient();
var groupPhoto = await hc.GetByteArrayAsync(
    "https://raw.githubusercontent.com/georg-jung/FaceAiSharp/master/examples/obama_family.jpg");
var img = Image.Load<Rgb24>(groupPhoto);

var det = FaceAiSharpBundleFactory.CreateFaceDetectorWithLandmarks();
var rec = FaceAiSharpBundleFactory.CreateFaceEmbeddingsGenerator();

var faces = det.DetectFaces(img);
var first = faces.First();
var second = faces.Skip(1).First();

// AlignFaceUsingLandmarks is an in-place operation so we need to create a clone of img first
var secondImg = img.Clone();
rec.AlignFaceUsingLandmarks(img, first.Landmarks!);
rec.AlignFaceUsingLandmarks(secondImg, second.Landmarks!);

var embedding1 = rec.GenerateEmbedding(img);
var embedding2 = rec.GenerateEmbedding(secondImg);

var dot = FaceAiSharp.Extensions.GeometryExtensions.Dot(embedding1, embedding2);

Console.WriteLine($"Dot product: {dot}");
if (dot >= 0.42)
{
    Console.WriteLine("Assessment: Both pictures show the same person.");
}
else if (dot > 0.28 && dot < 0.42)
{
    Console.WriteLine("Assessment: Hard to tell if the pictures show the same person.");
}
else if (dot <= 0.28)
{
    Console.WriteLine("Assessment: These are two different people.");
}
