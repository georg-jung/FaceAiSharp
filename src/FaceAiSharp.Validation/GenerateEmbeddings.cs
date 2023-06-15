// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.CommandLine;
using System.Data;
using System.Diagnostics;
using System.Threading.Channels;
using FaceAiSharp.Extensions;
using LiteDB;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp.Validation;

internal sealed class GenerateEmbeddings : IDisposable
{
    private readonly SessionOptions _embedderSessOpts;
    private readonly IFaceEmbeddingsGenerator _emb;
    private readonly IFaceDetector _det;
    private readonly IMemoryCache _cache;
    private readonly DirectoryInfo _dataset;
    private readonly FileInfo _db;
    private readonly string _dbEmbeddingCollectionName;
    private readonly FileInfo? _pairsFile;
    private readonly PreprocessingMode _preprocessingMode;

    public GenerateEmbeddings(DirectoryInfo dataset, FileInfo db, FileInfo arcFaceModel, FileInfo scrfdModel, string dbEmbeddingCollectionName, FileInfo? pairsFile, PreprocessingMode preprocessingMode = PreprocessingMode.AffineTransform)
    {
        _embedderSessOpts = new SessionOptions
        {
            EnableMemoryPattern = false,
        };
        _embedderSessOpts.AppendExecutionProvider_DML();

        _emb = new ArcFaceEmbeddingsGenerator(
            new()
            {
                ModelPath = arcFaceModel.FullName,
            },
            _embedderSessOpts);

        var opts = new MemoryCacheOptions();
        var iopts = Options.Create(opts);
        _cache = new MemoryCache(iopts);

        _det = new ScrfdDetector(
            _cache,
            new()
            {
                ModelPath = scrfdModel.FullName,
            });

        _dataset = dataset;
        _db = db;
        _dbEmbeddingCollectionName = dbEmbeddingCollectionName;
        _pairsFile = pairsFile;
        _preprocessingMode = preprocessingMode;
    }

    internal enum PreprocessingMode
    {
        /// <summary>
        /// Dont do any preprocessing. Just use the image as is.
        /// </summary>
        Noop,

        /// <summary>
        /// Just cut the face out of the image.
        /// </summary>
        Cut,

        /// <summary>
        /// Cut the face out of the image and rotate it so that both eyes are on the same horizontal line.
        /// </summary>
        Angle,

        /// <summary>
        /// Align the face using an affine transformation based on 5 facial landmark points.
        /// </summary>
        AffineTransform,
    }

    public async Task Invoke()
    {
        using var db = new LiteDatabase(_db.FullName);
        var dbEmb = db.GetCollection<Embedding>(_dbEmbeddingCollectionName);
        dbEmb.EnsureIndex(x => x.FilePath);

        var setFolder = _dataset.FullName;

        var setEnum = Dataset.EnumerateFolderPerIdentity(setFolder).Where(x => !dbEmb.Exists(db => db.FilePath == x.FilePath));
        if (_pairsFile is not null)
        {
            var pairsFiles = Dataset.CreatePairsBasedFileList(_pairsFile.FullName);
            setEnum = setEnum.Where(x => pairsFiles.Contains((x.Identity, x.ImageNumber)));
        }

        var setList = setEnum.ToList();

        Console.WriteLine($"Processing {setList.Count} images...");
        Console.WriteLine($"Preprocessing mode: {_preprocessingMode}");
        Console.WriteLine($"DB: {_db.FullName}");

        var ch = Channel.CreateBounded<ChannelData>(10);
        var producerTask = Task.Run(() => ProducePreprocessed(setList, ch));
        var embeddings = GenerateEmbeddingsFromChannel(ch);

        var cnt = 0;
        await foreach (var (embRes, ticks) in embeddings)
        {
            dbEmb.Insert(embRes);
            Console.WriteLine($"{cnt,5:D}/{setList.Count,5:D} {ticks,4:D}ms : {embRes.FilePath}");
            if (Interlocked.Increment(ref cnt) % 20 == 0)
            {
                db.Commit();
            }
        }

        db.Commit();

        await producerTask;
    }

    public void Dispose()
    {
        _embedderSessOpts.Dispose();
        (_emb as IDisposable)?.Dispose();
        (_det as IDisposable)?.Dispose();
        _cache.Dispose();
    }

    private Image<Rgb24> PreprocessAngle(string filePath)
    {
        var img = Image.Load<Rgb24>(filePath);
        var dets = _det.DetectFaces(img);
        var imgCenter = RectangleF.Center(img.Bounds());
        var middleFace = dets.MinBy(x => RectangleF.Center(x.Box).EuclideanDistance(imgCenter));
        Debug.Assert(middleFace.Landmarks != null, "No landmarks detected but required");
        var (leye, reye) = (ScrfdDetector.GetLeftEye(middleFace.Landmarks), ScrfdDetector.GetRightEye(middleFace.Landmarks));
        var angle = GeometryExtensions.GetAlignmentAngle(leye, reye);
        img.CropAlignedDestructive(Rectangle.Round(middleFace.Box), angle, 112);
        return img;
    }

    private Image<Rgb24> PreprocessCut(string filePath)
    {
        var img = Image.Load<Rgb24>(filePath);
        var dets = _det.DetectFaces(img);
        var imgCenter = RectangleF.Center(img.Bounds());
        var middleFace = dets.MinBy(x => RectangleF.Center(x.Box).EuclideanDistance(imgCenter));
        img.CropAlignedDestructive(Rectangle.Round(middleFace.Box), 0, 112);
        return img;
    }

    private Image<Rgb24> PreprocessNoop(string filePath)
    {
        var img = Image.Load<Rgb24>(filePath);
        return img;
    }

    private Image<Rgb24> PreprocessAffineTransform(string filePath)
    {
        var img = Image.Load<Rgb24>(filePath);
        var dets = _det.DetectFaces(img);
        var imgCenter = RectangleF.Center(img.Bounds());
        var middleFace = dets.MinBy(x => RectangleF.Center(x.Box).EuclideanDistance(imgCenter));
        Debug.Assert(middleFace.Landmarks != null, "No landmarks detected but required");
        _emb.AlignFaceUsingLandmarks(img, middleFace.Landmarks);
        return img;
    }

    private Image<Rgb24> Preprocess(string filePath)
    {
        return _preprocessingMode switch
        {
            PreprocessingMode.Noop => PreprocessNoop(filePath),
            PreprocessingMode.Cut => PreprocessCut(filePath),
            PreprocessingMode.Angle => PreprocessAngle(filePath),
            PreprocessingMode.AffineTransform => PreprocessAffineTransform(filePath),
            _ => throw new NotImplementedException(),
        };
    }

    private async Task ProducePreprocessed(IEnumerable<DatasetImage> images, ChannelWriter<ChannelData> channel)
    {
        await Parallel.ForEachAsync(images, async (DatasetImage image, CancellationToken tok) =>
        {
            var sw = Stopwatch.StartNew();
            var img = Preprocess(image.FilePath);
            await channel.WriteAsync(new(image, img, sw.ElapsedMilliseconds));
        });

        channel.Complete();
    }

    private async IAsyncEnumerable<(Embedding Result, long Ticks)> GenerateEmbeddingsFromChannel(ChannelReader<ChannelData> channel)
    {
        await foreach (var (metadata, img, ticks) in channel.ReadAllAsync())
        {
            var sw = Stopwatch.StartNew();
            var embedding = _emb.GenerateEmbedding(img);
            yield return (new Embedding() with
            {
                Identity = metadata.Identity,
                ImageNumber = metadata.ImageNumber,
                FilePath = metadata.FilePath,
                Embeddings = embedding,
            }, ticks + sw.ElapsedMilliseconds);
        }
    }

    internal readonly record struct ChannelData(DatasetImage Metadata, Image<Rgb24> Image, long ElapsedTicks);
}
