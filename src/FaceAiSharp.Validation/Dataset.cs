// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FaceAiSharp.Validation;

internal class Dataset
{
    public static IEnumerable<DatasetImage> EnumerateFolderPerIdentity(string parent, string searchPattern = "*.jpg")
    {
        var withoutSlash = Path.GetFullPath(parent);
        var cutoff = withoutSlash.Length + 1; // len with slash
        foreach (var file in Directory.EnumerateFiles(parent, searchPattern, SearchOption.AllDirectories))
        {
            var id = Path.GetDirectoryName(file)!.Substring(cutoff); // eg. John_Doe
            var fileNameOnly = Path.GetFileNameWithoutExtension(file)!; // eg. John_Doe_0001
            var imgNumStr = fileNameOnly.Substring(id.Length + 1); // eg. 0001
            var imgNum = Convert.ToInt32(imgNumStr);
            yield return new DatasetImage(id, imgNum, file);
        }
    }

    public static IEnumerable<IEnumerable<DefinedPair>> ParsePairs(string pairsFile)
    {
        using var sr = new StreamReader(pairsFile);
        var first = sr.ReadLine() ?? throw new ArgumentException("Wrong file format");
        var firstSplitted = first.Split('\t');
        var (folds, perFold) = (Convert.ToInt32(firstSplitted[0]), Convert.ToInt32(firstSplitted[1]));

        IEnumerable<DefinedPair> IterateFold()
        {
            for (var i = 0; i < perFold; i++)
            {
                var spl = (sr.ReadLine() ?? throw new ArgumentException("Wrong file format")).Split('\t');
                yield return new DefinedPair()
                {
                    Identity1 = spl[0],
                    ImageNumber1 = Convert.ToInt32(spl[1]),
                    ImageNumber2 = Convert.ToInt32(spl[2]),
                    SameIdentity = true,
                };
            }

            for (var i = 0; i < perFold; i++)
            {
                var spl = (sr.ReadLine() ?? throw new ArgumentException("Wrong file format")).Split('\t');
                yield return new DefinedPair()
                {
                    Identity1 = spl[0],
                    ImageNumber1 = Convert.ToInt32(spl[1]),
                    Identity2 = spl[2],
                    ImageNumber2 = Convert.ToInt32(spl[3]),
                    SameIdentity = false,
                };
            }
        }

        for (var iFold = 0; iFold < folds; iFold++)
        {
            yield return IterateFold();
        }
    }

    public static HashSet<(string Identity, int ImageNumber)> CreatePairsBasedFileList(string pairsFile)
    {
        var folds = ParsePairs(pairsFile);
        var pairs = folds.SelectMany(x => x);
        var ret = new HashSet<(string Identity, int ImageNumber)>();
        pairs.ForEach(p =>
        {
            var id2 = p.SameIdentity ? p.Identity1 : p.Identity2!;
            ret.Add((p.Identity1, p.ImageNumber1));
            ret.Add((id2, p.ImageNumber2));
        });

        return ret;
    }
}

[System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1201:Elements should appear in the correct order", Justification = "I like it here")]
internal readonly record struct DatasetImage(string Identity, int ImageNumber, string FilePath);
