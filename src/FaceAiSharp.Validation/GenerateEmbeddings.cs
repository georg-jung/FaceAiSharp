// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.CommandLine;
using System.Data;
using System.Diagnostics;
using System.Threading.Channels;
using FaceAiSharp.Abstractions;
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

    public GenerateEmbeddings(DirectoryInfo dataset, FileInfo db, FileInfo arcFaceModel, FileInfo scrfdModel, string dbEmbeddingCollectionName, FileInfo? pairsFile)
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

        var ch = Channel.CreateBounded<ChannelData>(10);
        var producerTask = Task.Run(() => ProducePreprocessed(setEnum, ch));
        var embeddings = GenerateEmbeddingsFromChannel(ch);

        await foreach (var (embRes, ticks) in embeddings)
        {
            dbEmb.Insert(embRes);
            Console.WriteLine($"{ticks,4:D}ms : {embRes.FilePath}");
            db.Commit();
        }

        await producerTask;
    }

    public void Dispose()
    {
        _embedderSessOpts.Dispose();
        (_emb as IDisposable)?.Dispose();
        (_det as IDisposable)?.Dispose();
        _cache.Dispose();
    }

    private Image<Rgb24> Preprocess_Angle(string filePath)
    {
        var img = Image.Load<Rgb24>(filePath);
        var x = _det.Detect(img);
        var first = x.First();
        Debug.Assert(first.Landmarks != null, "No landmarks detected but required");
        var angle = ScrfdDetector.GetFaceAlignmentAngle(first.Landmarks);
        img.CropAlignedDestructive(Rectangle.Round(first.Box), (float)angle);
        return img;
    }

    private Image<Rgb24> Preprocess(string filePath)
    {
        var img = Image.Load<Rgb24>(filePath);
        var x = _det.Detect(img);
        var first = x.First();
        Debug.Assert(first.Landmarks != null, "No landmarks detected but required");
        ArcFaceEmbeddingsGenerator.AlignUsingFacialLandmarks(img, first.Landmarks);
        return img;
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
            var embedding = _emb.Generate(img);
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
