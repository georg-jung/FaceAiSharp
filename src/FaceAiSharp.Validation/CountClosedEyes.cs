// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using FaceAiSharp.Extensions;
using LiteDB;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp.Validation;

internal class CountClosedEyes
{
    private readonly FileInfo _db;
    private readonly DirectoryInfo _dataset;
    private readonly SessionOptions _dmlSessOpts;
    private readonly MemoryCache _cache;
    private readonly ScrfdDetector _det;
    private readonly OpenVinoOpenClosedEye0001 _eyeState;

    public CountClosedEyes(DirectoryInfo dataset, FileInfo db, FileInfo scrfdModel, FileInfo eyeStateModel)
    {
        _db = db;
        _dataset = dataset;

        _dmlSessOpts = new SessionOptions
        {
            EnableMemoryPattern = false,
        };
        _dmlSessOpts.AppendExecutionProvider_DML();

        var opts = new MemoryCacheOptions();
        var iopts = Options.Create(opts);
        _cache = new MemoryCache(iopts);

        _det = new ScrfdDetector(
            _cache,
            new()
            {
                ModelPath = scrfdModel.FullName,
            });

        _eyeState = new OpenVinoOpenClosedEye0001(
            new OpenVinoOpenClosedEye0001Options()
            {
                ModelPath = eyeStateModel.FullName,
            });
    }

    public void Invoke()
    {
        using var db = new LiteDatabase(_db.FullName);
        var dbES = db.GetCollection<EyeState>("EyeState");
        dbES.DeleteAll();
        dbES.EnsureIndex(x => x.FilePath, true);

        var detTicks = 0L;
        var esTicks = 0L;
        var faces = 0;
        var files = 0;
        foreach (var file in _dataset.EnumerateFiles("*.jpg"))
        {
            using var img = Image.Load<Rgb24>(file.FullName);
            var sw = Stopwatch.StartNew();
            var x = _det.DetectFaces(img);
            detTicks += sw.ElapsedTicks;
            int open = 0;
            int closed = 0;
            foreach (var face in x)
            {
                faces++;
                var lmrks = face.Landmarks ?? throw new InvalidOperationException();
                var (leye, reye) = (ScrfdDetector.GetLeftEye(lmrks), ScrfdDetector.GetRightEye(lmrks));
                var angle = GeometryExtensions.GetAlignmentAngle(leye, reye);
                var dist = leye.EuclideanDistance(reye);
                var squareAroundEyeLen = dist / 3;

                if (squareAroundEyeLen < 8)
                {
                    continue;
                }

                var eyeRectSz = new Size((int)squareAroundEyeLen * 2);
                leye.Offset(-squareAroundEyeLen, -squareAroundEyeLen);
                reye.Offset(-squareAroundEyeLen, -squareAroundEyeLen);
                var leyeRect = new Rectangle(Point.Round(leye), eyeRectSz);
                var reyeRect = new Rectangle(Point.Round(reye), eyeRectSz);
                var leyeImg = img.CropAligned(leyeRect, angle, 32);
                var reyeImg = img.CropAligned(reyeRect, angle, 32);

                sw.Restart();
                var leftOpen = _eyeState.IsOpen(leyeImg);
                var rightOpen = _eyeState.IsOpen(reyeImg);
                esTicks += sw.ElapsedTicks;
                open += leftOpen ? 1 : 0;
                closed += !leftOpen ? 1 : 0;
                open += rightOpen ? 1 : 0;
                closed += !rightOpen ? 1 : 0;
            }

            dbES.Upsert(new EyeState()
            {
                FilePath = file.FullName,
                ClosedEyes = closed,
                OpenEyes = open,
            });

            files++;
        }

        Console.WriteLine($"Files:           {files}");
        Console.WriteLine($"Faces:           {faces}");
        Console.WriteLine($"Detection ticks: {detTicks}");
        Console.WriteLine($"Eye state ticks: {esTicks}");
    }
}
