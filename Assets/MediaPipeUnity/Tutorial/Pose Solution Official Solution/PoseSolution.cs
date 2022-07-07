using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Mediapipe.Unity.CoordinateSystem;

using Stopwatch = System.Diagnostics.Stopwatch;

namespace Mediapipe.Unity
{
  public class PoseSolution : MonoBehaviour
  {
    [SerializeField] private TextAsset _configAsset;
    [SerializeField] private RawImage _screen;
    [SerializeField] private int _width;
    [SerializeField] private int _height;
    [SerializeField] private int _fps;

    public bool smoothLandmarks = true;
    public bool enableSegmentation = true;
    public bool smoothSegmentation = true;

    private CalculatorGraph _graph;
    private ResourceManager _resourceManager;

    private WebCamTexture _webCamTexture;
    private Texture2D _inputTexture;
    private Color32[] _inputPixelData;
    private Texture2D _outputTexture;
    private Color32[] _outputPixelData;

    private const string _InputStreamName = "input_video";
    private const string _PoseDetectionStreamName = "pose_detection";
    private const string _PoseLandmarksStreamName = "pose_landmarks";
    private const string _PoseWorldLandmarksStreamName = "pose_world_landmarks";
    private const string _SegmentationMaskStreamName = "segmentation_mask";
    private const string _RoiFromLandmarksStreamName = "roi_from_landmarks";

    private OutputStream<DetectionPacket, Detection> _poseDetectionStream;
    private OutputStream<NormalizedLandmarkListPacket, NormalizedLandmarkList> _poseLandmarksStream;
    private OutputStream<LandmarkListPacket, LandmarkList> _poseWorldLandmarksStream;
    private OutputStream<ImageFramePacket, ImageFrame> _segmentationMaskStream;
    private OutputStream<NormalizedRectPacket, NormalizedRect> _roiFromLandmarksStream;

    public RotationAngle rotation { get; private set; } = 0;

    public enum ModelComplexity
    {
      Lite = 0,
      Full = 1,
      Heavy = 2,
    }
    public ModelComplexity modelComplexity = ModelComplexity.Full;

    public void RequestDependentAssets()
    {
        AssetLoader.Provide(new StreamingAssetsResourceManager());
        AssetLoader.PrepareAssetAsync("pose_detection.bytes", "pose_detection.bytes", false);

        switch (modelComplexity)
        {
          case ModelComplexity.Lite: AssetLoader.PrepareAssetAsync("pose_landmark_lite.bytes", "pose_landmark_lite.bytes", false); break;
          case ModelComplexity.Full: AssetLoader.PrepareAssetAsync("pose_landmark_full.bytes", "pose_landmark_full.bytes", false); break;
          case ModelComplexity.Heavy: AssetLoader.PrepareAssetAsync("pose_landmark_heavy.bytes", "pose_landmark_heavy.bytes", false); break;
          default: throw new InternalException($"Invalid model complexity: {modelComplexity}");
        }
    }

    private IEnumerator Start()
    {
      if (WebCamTexture.devices.Length == 0)
      {
        throw new System.Exception("Web Camera devices are not found");
      }
      var webCamDevice = WebCamTexture.devices[0];
      _webCamTexture = new WebCamTexture(webCamDevice.name, _width, _height, _fps);
      _webCamTexture.Play();

      yield return new WaitUntil(() => _webCamTexture.width > 16);

      _screen.rectTransform.sizeDelta = new Vector2(_width, _height);

      _inputTexture = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
      _inputPixelData = new Color32[_width * _height];
      _outputTexture = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
      _outputPixelData = new Color32[_width * _height];

      _screen.texture = _outputTexture;

      _graph = new CalculatorGraph(_configAsset.text);
      _graph.StartRun().AssertOk();

      // request Dependant Assets
      RequestDependentAssets();

      var stopwatch = new Stopwatch();
      stopwatch.Start();

      _poseDetectionStream.StartPolling().AssertOk();
      _poseLandmarksStream.StartPolling().AssertOk();
      _poseWorldLandmarksStream.StartPolling().AssertOk();
      _segmentationMaskStream.StartPolling().AssertOk();
      _roiFromLandmarksStream.StartPolling().AssertOk();

      _graph.StartRun(BuildSidePacket()).AssertOk();

    }

    private void OnDestroy()
    {
      if (_webCamTexture != null)
      {
        _webCamTexture.Stop();
      }

      if (_graph != null)
      {
        try
        {
          _graph.CloseInputStream("input_video").AssertOk();
          _graph.WaitUntilDone().AssertOk();
        }
        finally
        {

          _graph.Dispose();
        }
      }
    }

    protected void SetImageTransformationOptions(SidePacket sidePacket, bool expectedToBeMirrored = false)
    {
      // NOTE: The origin is left-bottom corner in Unity, and right-top corner in MediaPipe.
      rotation = rotation.Reverse();
      var inputRotation = rotation;
      var isInverted = CoordinateSystem.ImageCoordinate.IsInverted(rotation);
      var shouldBeMirrored = false ^ expectedToBeMirrored;
      var inputHorizontallyFlipped = isInverted ^ shouldBeMirrored;
      var inputVerticallyFlipped = !isInverted;

      if ((inputHorizontallyFlipped && inputVerticallyFlipped) || rotation == RotationAngle.Rotation180)
      {
        inputRotation = inputRotation.Add(RotationAngle.Rotation180);
        inputHorizontallyFlipped = !inputHorizontallyFlipped;
        inputVerticallyFlipped = !inputVerticallyFlipped;
      }

      Logger.LogDebug($"input_rotation = {inputRotation}, input_horizontally_flipped = {inputHorizontallyFlipped}, input_vertically_flipped = {inputVerticallyFlipped}");

      sidePacket.Emplace("input_rotation", new IntPacket((int)inputRotation));
      sidePacket.Emplace("input_horizontally_flipped", new BoolPacket(inputHorizontallyFlipped));
      sidePacket.Emplace("input_vertically_flipped", new BoolPacket(inputVerticallyFlipped));
    }

    private SidePacket BuildSidePacket()
    {
      var sidePacket = new SidePacket();

      SetImageTransformationOptions(sidePacket);

      // TODO: refactoring
      // The orientation of the output image must match that of the input image.
      var isInverted = CoordinateSystem.ImageCoordinate.IsInverted(rotation);
      var outputRotation = rotation;
      var outputHorizontallyFlipped = !isInverted && false;
      var outputVerticallyFlipped = (!true && false) ^ (isInverted && false);

      if ((outputHorizontallyFlipped && outputVerticallyFlipped) || outputRotation == RotationAngle.Rotation180)
      {
        outputRotation = outputRotation.Add(RotationAngle.Rotation180);
        outputHorizontallyFlipped = !outputHorizontallyFlipped;
        outputVerticallyFlipped = !outputVerticallyFlipped;
      }

      sidePacket.Emplace("output_rotation", new IntPacket((int)outputRotation));
      sidePacket.Emplace("output_horizontally_flipped", new BoolPacket(outputHorizontallyFlipped));
      sidePacket.Emplace("output_vertically_flipped", new BoolPacket(outputVerticallyFlipped));

      Logger.LogDebug($"output_rotation = {outputRotation}, output_horizontally_flipped = {outputHorizontallyFlipped}, output_vertically_flipped = {outputVerticallyFlipped}");

      sidePacket.Emplace("model_complexity", new IntPacket((int)modelComplexity));
      sidePacket.Emplace("smooth_landmarks", new BoolPacket(smoothLandmarks));
      sidePacket.Emplace("enable_segmentation", new BoolPacket(enableSegmentation));
      sidePacket.Emplace("smooth_segmentation", new BoolPacket(smoothSegmentation));

      return sidePacket;
    }
  }
}
