// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using FaceAiSharp.Extensions;
using Shouldly;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp.Tests;

public class ArcFaceEmbeddingsGeneratorTests
    : IClassFixture<ArcFaceEmbeddingsGeneratorFixture>,
        IClassFixture<ScrfdDetectorFixture>
{
    private readonly ArcFaceEmbeddingsGenerator _emb;
    private readonly ScrfdDetector _det;

    public ArcFaceEmbeddingsGeneratorTests(ArcFaceEmbeddingsGeneratorFixture arcFaceFixture, ScrfdDetectorFixture detFixture)
    {
        _emb = arcFaceFixture.Embedder;
        _det = detFixture.Detector;
    }

    [Fact]
    public async Task SamePictureTwice()
    {
        var x = await EmbedObama3();
        var y = await EmbedObama3();
        x.Dot(y).ShouldBeGreaterThan(0.99f);
    }

    [Fact]
    public async Task ObamaIsPartOfHisFamilyPhoto()
    {
        var obama = await EmbedObama3();
        var family = await EmbedObamaFamily();
        family.Select(x => x.Dot(obama)).Max().ShouldBeGreaterThan(0.42f);

        // but all the others on the picture are different persons
        var cartesian = from x in family
                        from y in family
                        select (First: x, Second: y);
        cartesian.Where(x => x.First != x.Second).ShouldAllBe(x => x.First.Dot(x.Second) < 0.4f);
    }

    private async Task<float[]> EmbedObama3()
    {
        using var img = await Image.LoadAsync<Rgb24>("TestData/jpgs/Barack_Obama_03.jpg");
        var res = _det.DetectFaces(img);
        ArcFaceEmbeddingsGenerator.AlignFaceUsingLandmarks(img, res.Single().Landmarks ?? throw new InvalidOperationException("Landmarks required."));

        return _emb.GenerateEmbedding(img);
    }

    private async Task<List<float[]>> EmbedObamaFamily()
    {
        using var img1 = await Image.LoadAsync<Rgb24>("TestData/jpgs/obama_family.jpg");
        using var img2 = img1.Clone();
        using var img3 = img1.Clone();
        using var img4 = img1.Clone();

        var res = _det.DetectFaces(img1).ToList();
        ArcFaceEmbeddingsGenerator.AlignFaceUsingLandmarks(img1, res[0].Landmarks ?? throw new InvalidOperationException("Landmarks required."));
        ArcFaceEmbeddingsGenerator.AlignFaceUsingLandmarks(img2, res[1].Landmarks ?? throw new InvalidOperationException("Landmarks required."));
        ArcFaceEmbeddingsGenerator.AlignFaceUsingLandmarks(img3, res[2].Landmarks ?? throw new InvalidOperationException("Landmarks required."));
        ArcFaceEmbeddingsGenerator.AlignFaceUsingLandmarks(img4, res[3].Landmarks ?? throw new InvalidOperationException("Landmarks required."));

        var emb1 = _emb.GenerateEmbedding(img1);
        var emb2 = _emb.GenerateEmbedding(img2);
        var emb3 = _emb.GenerateEmbedding(img3);
        var emb4 = _emb.GenerateEmbedding(img4);
        return [emb1, emb2, emb3, emb4];
    }
}
