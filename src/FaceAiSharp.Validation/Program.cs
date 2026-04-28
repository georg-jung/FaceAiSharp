// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.CommandLine;
using FaceAiSharp.Validation;

var rc = new RootCommand("FaceAiSharp validation tools");

var db = new Option<FileInfo>("--db")
{
    Description = "File to use as db to store results and for continuation",
    DefaultValueFactory = _ => new FileInfo("faceaisharp-validation.litedb"),
    Recursive = true,
};
rc.Options.Add(db);

var dbEmbeddingCollectionName = new Option<string>("---db-embedding-collection-name")
{
    DefaultValueFactory = _ => "ArcfaceEmbeddings",
};

var dataset = new Option<DirectoryInfo>("--dataset")
{
    DefaultValueFactory = _ => new DirectoryInfo(@"C:\Users\georg\Downloads\lfw\lfw"),
};

var pairsFile = new Option<FileInfo>("--pairs-file")
{
    DefaultValueFactory = _ => new FileInfo(@"C:\Users\georg\Downloads\lfw\pairs.txt"),
};

var arcfaceModel = new Option<FileInfo>("--arcface-model")
{
    DefaultValueFactory = _ => new FileInfo(@"C:\Users\georg\facePics\arcfaceresnet100-8\resnet100\resnet100.onnx"),
};

var scrfdModel = new Option<FileInfo>("--scrfd-model")
{
    DefaultValueFactory = _ => new FileInfo(@"C:\Users\georg\OneDrive\Dokumente\BlazorFace\ScrfdOnnx\scrfd_2.5g_bnkps.onnx"),
};

var eyeStateModel = new Option<FileInfo>("--eyestate-model")
{
    DefaultValueFactory = _ => new FileInfo(@"C:\Users\georg\OneDrive\Dokumente\BlazorFace\openvino_open-closed-eye-0001\open_closed_eye.onnx"),
};

var threshold = new Option<float>("--threshold")
{
    DefaultValueFactory = _ => 0.29f,
};

var binJpegs = new Option<DirectoryInfo>("--bin-jpegs")
{
    Required = true,
};

var preprocMode = new Option<GenerateEmbeddings.PreprocessingMode>("--prprocessing-mode")
{
    DefaultValueFactory = _ => GenerateEmbeddings.PreprocessingMode.AffineTransform,
};

var generateEmbeddings = new Command("generate-embeddings") { dataset, arcfaceModel, scrfdModel, dbEmbeddingCollectionName, pairsFile, preprocMode };

var calcAllDistances = new Command("calc-all-distances") { dbEmbeddingCollectionName, threshold };

var calcPairsDistances = new Command("calc-pairs-distances") { dbEmbeddingCollectionName, threshold, pairsFile };

var countClosedEyes = new Command("count-closed-eyes") { dataset, scrfdModel, eyeStateModel };

var renameModelzooBinJpegs = new Command("rename-modelzoo-bin-jpegs") { binJpegs, pairsFile };

#pragma warning disable SA1116 // Split parameters should start on line after declaration
#pragma warning disable SA1117 // Parameters should be on same line or separate lines

generateEmbeddings.SetAction(async (ParseResult parseResult, CancellationToken cancellationToken) =>
{
    using var cmd = new GenerateEmbeddings(
        parseResult.GetRequiredValue(dataset),
        parseResult.GetRequiredValue(db),
        parseResult.GetRequiredValue(arcfaceModel),
        parseResult.GetRequiredValue(scrfdModel),
        parseResult.GetRequiredValue(dbEmbeddingCollectionName),
        parseResult.GetRequiredValue(pairsFile),
        parseResult.GetRequiredValue(preprocMode));
    await cmd.Invoke();
    return 0;
});
rc.Subcommands.Add(generateEmbeddings);

calcAllDistances.SetAction((ParseResult parseResult, CancellationToken cancellationToken) =>
{
    var calc = new CalculateAllDistances(
        parseResult.GetRequiredValue(db),
        parseResult.GetRequiredValue(dbEmbeddingCollectionName),
        parseResult.GetRequiredValue(threshold));
    calc.Invoke();
    return Task.FromResult(0);
});
rc.Subcommands.Add(calcAllDistances);

calcPairsDistances.SetAction((ParseResult parseResult, CancellationToken cancellationToken) =>
{
    var calc = new CalculatePairsDistances(
        parseResult.GetRequiredValue(db),
        parseResult.GetRequiredValue(dbEmbeddingCollectionName),
        parseResult.GetRequiredValue(threshold),
        parseResult.GetRequiredValue(pairsFile));
    calc.Invoke();
    return Task.FromResult(0);
});
rc.Subcommands.Add(calcPairsDistances);

countClosedEyes.SetAction((ParseResult parseResult, CancellationToken cancellationToken) =>
{
    var cnt = new CountClosedEyes(
        parseResult.GetRequiredValue(dataset),
        parseResult.GetRequiredValue(db),
        parseResult.GetRequiredValue(scrfdModel),
        parseResult.GetRequiredValue(eyeStateModel));
    cnt.Invoke();
    return Task.FromResult(0);
});
rc.Subcommands.Add(countClosedEyes);

renameModelzooBinJpegs.SetAction((ParseResult parseResult, CancellationToken cancellationToken) =>
{
    var ren = new RenameModelzooBinJpegs(
        parseResult.GetRequiredValue(binJpegs),
        parseResult.GetRequiredValue(pairsFile));
    ren.Invoke();
    return Task.FromResult(0);
});
rc.Subcommands.Add(renameModelzooBinJpegs);

return await rc.Parse(args).InvokeAsync();
