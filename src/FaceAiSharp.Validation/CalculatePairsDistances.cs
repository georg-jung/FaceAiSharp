// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using FaceAiSharp.Extensions;
using Humanizer;
using LiteDB;

namespace FaceAiSharp.Validation;

internal class CalculatePairsDistances
{
    private readonly FileInfo _db;
    private readonly string _dbEmbeddingCollectionName;
    private readonly float _threshold;
    private readonly FileInfo _pairsFile;

    public CalculatePairsDistances(FileInfo db, string dbEmbeddingCollectionName, float threshold, FileInfo pairsFile)
    {
        _db = db;
        _dbEmbeddingCollectionName = dbEmbeddingCollectionName;
        _threshold = threshold;
        _pairsFile = pairsFile;
    }

    public void Invoke()
    {
        DoWork(_threshold);
    }

    private void DoWork(float threshold, bool shortOutput = false)
    {
        var sw = Stopwatch.StartNew();
        using var db = new LiteDatabase(_db.FullName);
        var dbEmb = db.GetCollection<Embedding>(_dbEmbeddingCollectionName);
        dbEmb.EnsureIndex(x => x.Identity);

        var pairs = Dataset.ParsePairs(_pairsFile.FullName).SelectMany(x => x);

        double avgCosDist = 0;
        double avgEuclDist = 0;
        double avgDotProd = 0;
        double avgCosDistTrue = 0;
        double avgEuclDistTrue = 0;
        double avgDotProdTrue = 0;
        double avgCosDistFalse = 0;
        double avgEuclDistFalse = 0;
        double avgDotProdFalse = 0;
        int cnt = 0;
        int trueCnt = 0;
        int missing = 0;

        var tp = 0;
        var tn = 0;
        var fp = 0;
        var fn = 0;
        var confidences = new List<(float Confidence, bool IsMatch)>(6000); // 6000 = number of pairs in pairs.txt of lfw

        void ReportProgressLong()
        {
            confidences.Sort((x, y) => y.Confidence.CompareTo(x.Confidence)); // we want to sort desc so exchange x and y
            Console.WriteLine($"Calculated {cnt} pairs.");
            if (missing > 0)
            {
                Console.WriteLine($"{missing} pairs MISSING!");
            }

            Console.WriteLine($"Had {trueCnt} pairs belonging to the same person.");
            Console.WriteLine($"Avergage cosine distance:                   {avgCosDist / cnt}");
            Console.WriteLine($"Avergage euclidean distance:                {avgEuclDist / cnt}");
            Console.WriteLine($"Avergage dot product:                       {avgDotProd / cnt}");
            Console.WriteLine($"Avergage cosine distance [same person]:     {avgCosDistTrue / trueCnt}");
            Console.WriteLine($"Avergage euclidean distance [same person]:  {avgEuclDistTrue / trueCnt}");
            Console.WriteLine($"Avergage dot product [same person]:         {avgDotProdTrue / trueCnt}");
            Console.WriteLine($"Avergage cosine distance [diff. person]:    {avgCosDistFalse / (cnt - trueCnt)}");
            Console.WriteLine($"Avergage euclidean distance [diff. person]: {avgEuclDistFalse / (cnt - trueCnt)}");
            Console.WriteLine($"Avergage dot product [diff. person]:        {avgDotProdFalse / (cnt - trueCnt)}");
            Console.WriteLine($"P: {tp + fn,8:D} TP: {tp,8:D} FP: {fp,8:D}");
            Console.WriteLine($"N: {tn + fp,8:D} TN: {tn,8:D} FN: {fn,8:D}");
            Console.WriteLine($"Accuracy:  {(double)(tp + tn) / (tp + tn + fp + fn):P2}");
            Console.WriteLine($"Precision: {(double)tp / (tp + fp):P2}");
            Console.WriteLine($"Recall:    {(double)tp / (tp + fn):P2}");
            Console.WriteLine($"F1 score:  {(double)tp * 2 / ((tp * 2) + fp + fn):N4}");
            Console.WriteLine($"AuROC:     {Metrics.Auc(confidences)}");
            Console.WriteLine($"Threshold for best accuracy: {Metrics.FindThreshold(confidences)}");
            Console.WriteLine();
        }

        void ReportProgressShort()
        {
            Console.WriteLine($"Thresh:\t{threshold}\tAcc:{(double)(tp + tn) / (tp + tn + fp + fn):P2}\tF1:\t{(double)tp * 2 / ((tp * 2) + fp + fn):N4}");
        }

        foreach (var pair in pairs)
        {
            var embX = dbEmb.FindOne(db => db.Identity == pair.Identity1 && db.ImageNumber == pair.ImageNumber1);
            var yId = pair.SameIdentity ? pair.Identity1 : pair.Identity2;
            var embY = dbEmb.FindOne(db => db.Identity == yId && db.ImageNumber == pair.ImageNumber2);
            if (embX is null || embY is null)
            {
                missing++;
                continue;
            }

            var cosDist = embX.Embeddings.CosineDistance(embY.Embeddings);
            var euclDist = embX.Embeddings.EuclideanDistance(embY.Embeddings);
            var dotProd = embX.Embeddings.Dot(embY.Embeddings);
            var sameIdnt = pair.SameIdentity;
            avgCosDist += cosDist;
            avgEuclDist += euclDist;
            avgDotProd += dotProd;
            confidences.Add((dotProd, sameIdnt));
            if (sameIdnt)
            {
                avgCosDistTrue += cosDist;
                avgEuclDistTrue += euclDist;
                avgDotProdTrue += dotProd;
                trueCnt++;
            }
            else
            {
                avgCosDistFalse += cosDist;
                avgEuclDistFalse += euclDist;
                avgDotProdFalse += dotProd;
            }

#pragma warning disable SA1503 // Braces should not be omitted
            if (dotProd > threshold && sameIdnt) tp++;
            if (dotProd <= threshold && !sameIdnt) tn++;
#pragma warning restore SA1503 // Braces should not be omitted
            if (dotProd > threshold && !sameIdnt)
            {
                fp++;
                Console.WriteLine($"FP:\t{pair.Identity1}\t{pair.ImageNumber1}\t{pair.Identity2}\t{pair.ImageNumber2}\t{dotProd}");
            }

            if (dotProd <= threshold && sameIdnt)
            {
                fn++;
                Console.WriteLine($"FN:\t{pair.Identity1}\t{pair.ImageNumber1}\t{pair.Identity2 ?? pair.Identity1}\t{pair.ImageNumber2}\t{dotProd}");
            }

            cnt++;
        }

        if (shortOutput)
        {
            ReportProgressShort();
        }
        else
        {
            Console.WriteLine($"--- FINISHED! Final Stats {threshold} ---");
            ReportProgressLong();
            Console.WriteLine($"Took {sw.Elapsed.Humanize(4)}");
        }
    }
}
