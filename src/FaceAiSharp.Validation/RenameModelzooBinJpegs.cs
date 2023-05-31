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

internal sealed class RenameModelzooBinJpegs
{
    private readonly DirectoryInfo _binJpegs;
    private readonly FileInfo _pairsFile;

    public RenameModelzooBinJpegs(DirectoryInfo binJpegs, FileInfo pairsFile)
    {
        _binJpegs = binJpegs;
        _pairsFile = pairsFile;
    }

    public void Invoke()
    {
        var fld = _binJpegs.FullName;
        var pairs = Dataset.ParsePairs(_pairsFile.FullName).SelectMany(x => x);
        var cnt = 0;
        foreach (var defPair in pairs)
        {
            var leftFile = Path.Combine(fld, $"{cnt * 2:D5}.jpg");
            var rightFile = Path.Combine(fld, $"{(cnt * 2) + 1:D5}.jpg");
            var fldLeft = Path.Combine(fld, defPair.Identity1);
            string id2 = defPair.SameIdentity ? defPair.Identity1 : defPair.Identity2!;
            var fldRight = Path.Combine(fld, id2);
            Directory.CreateDirectory(fldLeft);
            Directory.CreateDirectory(fldRight);

            var leftNew = Path.Combine(fldLeft, defPair.Identity1 + $"_{defPair.ImageNumber1:D4}.jpg");
            var rightNew = Path.Combine(fldRight, id2 + $"_{defPair.ImageNumber2:D4}.jpg");
            File.Move(leftFile, leftNew, true);
            File.Move(rightFile, rightNew, true);
            cnt++;
        }
    }
}
