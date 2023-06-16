FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src

RUN dotnet new console && \
    dotnet add package Microsoft.ML.OnnxRuntime && \
    dotnet add package FaceAiSharp.Bundle

COPY "Program.cs" .

RUN dotnet run
