# FaceAiSharp

FaceAiSharp is a .NET library that allows you to work with face-related computer vision tasks. It currently provides face detection, face recognition, facial landmarks detection, and eye state detection functionalities. FaceAiSharp leverages publicly available pretrained ONNX models to deliver accurate and efficient results and offers a convenient way to integrate them into your .NET applications. Whether you need to find faces, recognize individuals, detect facial landmarks, or determine eye states, FaceAiSharp simplifies the process with its simple API. ONNXRuntime is used for model inference, enabling hardware acceleration were possible.

## Features

- **Face Detection**: Identifies the boundaries of faces within images.
- **Face Recognition**: Recognizes and distinguishes between different faces.
- **Facial Landmarks Detection**: Detect and extract key facial landmarks such as eye centers, nose tip, and corners of the mouth.
- **Eye State Detection**: Determine whether eyes are open or closed.

## Key Highlights

- Utilizes publicly available, state-of-the-art, pretrained ONNX models to ensure accuracy and performance.
- All processing is done locally, with no reliance on cloud services.
- Supports image-based face processing using ImageSharp.
- Provides a simple and intuitive API for easy integration into your applications.
- Cross-platform support for Windows, Linux, Android and more.

## Getting Started

1. Create a new console app `dotnet new console`
2. Install two packages providing the relevant code and models:

    ```shell
    dotnet add package Microsoft.ML.OnnxRuntime
    dotnet add package FaceAiSharp.Bundle
    ```

3. Replace the content of the Program.cs file with the following code:

    ```csharp
    using FaceAiSharp;
    using FaceAiSharp.Extensions;
    using SixLabors.ImageSharp;
    using SixLabors.ImageSharp.PixelFormats;

    using var hc = new HttpClient();
    var groupPhoto = await hc.GetByteArrayAsync(
        "https://raw.githubusercontent.com/georg-jung/FaceAiSharp/master/examples/obama_family.jpg");
    var img = Image.Load<Rgb24>(groupPhoto);

    var det = FaceAiSharpBundleFactory.CreateFaceDetectorWithLandmarks();
    var rec = FaceAiSharpBundleFactory.CreateFaceEmbeddingsGenerator();

    var faces = det.DetectFaces(img);

    foreach (var face in faces)
    {
        Console.WriteLine($"Found a face with conficence {face.Confidence}: {face.Box}");
    }

    var first = faces.First();
    var second = faces.Skip(1).First();

    // AlignFaceUsingLandmarks is an in-place operation so we need to create a clone of img first
    var secondImg = img.Clone();
    rec.AlignFaceUsingLandmarks(img, first.Landmarks!);
    rec.AlignFaceUsingLandmarks(secondImg, second.Landmarks!);

    img.Save("aligned.jpg");
    Console.WriteLine($"Saved an aligned version of one of the faces found at \"./aligned.jpg\".");

    var embedding1 = rec.GenerateEmbedding(img);
    var embedding2 = rec.GenerateEmbedding(secondImg);

    var dot = embedding1.Dot(embedding2);

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

    ```

4. Run the program you just created: `dotnet run`

## See FaceAiSharp in Action

Try the interactive examples on <https://facerec.gjung.com/> or review the code behind the app: <https://github.com/georg-jung/explain-face-rec>

## Models

- Face Detection and Facial Landmarks Detection: [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)_2.5G_KPS ![Nuget](https://img.shields.io/nuget/v/FaceAiSharp.Models.Scrfd.2dot5g_kps?color=blue)
  - WIDERFACE Hard/Medium/Easy accuracy: 93.80% / 92.02% / 77.13%
  - GFlops: 2.5
  - Model size: 3,215 KB
- Face Recognition: [ArcFace from ONNX Model Zoo](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface)
  - fp32 ![Nuget](https://img.shields.io/nuget/v/FaceAiSharp.Models.ArcFace.LResNet100E-IR?color=blue)
    - LFW Accuracy: 99.77%
    - Model size: 248.9 MB
  - int8 quantized ![Nuget](https://img.shields.io/nuget/v/FaceAiSharp.Models.ArcFace.LResNet100E-IR-int8?color=blue)
    - LFW accuracy: 99.80%
    - Model size: 63 MB
    - Depending on your specific hardware, inference might be more or less efficient than with the fp32 variant.
- Eye State Detection: Intel OpenVINO's [open-closed-eye-0001](https://docs.openvino.ai/2022.3/omz_models_model_open_closed_eye_0001.html) ![Nuget](https://img.shields.io/nuget/v/FaceAiSharp.Models.OpenVino.open-closed-eye-0001?color=blue)
  - Accuracy: 95.85%
  - GFlops: 0.0014
  - Model size: 46 KB

See also: <https://github.com/georg-jung/FaceAiSharp.Models>

## License

FaceAiSharp is released under the MIT License - see the [LICENSE](LICENSE) file for details. The pretrained ONNX models used by FaceAiSharp are subject to their respective licenses. Please refer to the license terms provided by the original creators of the pretrained models for more information. These licenses may be more restrictive than the MIT license. The fact that MIT License is specified in the repository does not mean that the models or other third-party works are made available under the license.

## Contributing

Contributions to FaceAiSharp are always welcome! If you find a bug, have a feature request, or want to contribute code or documentation improvements, please open an issue or submit a pull request.
