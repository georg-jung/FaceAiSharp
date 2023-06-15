// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp;

public interface IEyeStateDetector
{
    /// <summary>
    ///     Determines if the eye represented in the given image is open.
    /// </summary>
    /// <param name="eyeImage">
    ///     An image of an eye. The image should be cropped to contain only the region of interest (the eye).
    /// </param>
    /// <returns>
    ///     Returns 'true' if the eye in the image is determined to be open, and 'false' if it is closed.
    /// </returns>
    bool IsOpen(Image<Rgb24> eyeImage);
}
