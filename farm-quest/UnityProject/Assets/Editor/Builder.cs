// Headless build entrypoint. Invoke with:
//
//   Unity -batchmode -nographics -projectPath <UnityProject> \
//         -executeMethod TeleopDataCollector.Builder.BuildAndroid \
//         -logFile <log> -quit
//
// Configures XR + OpenXR + Quest, synthesizes the scene, builds the APK.

using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace TeleopDataCollector
{
    public static class Builder
    {
        const string PACKAGE_ID    = "com.farm.quest";
        const string PRODUCT_NAME  = "FARM Quest";
        const string COMPANY_NAME  = "FARM";
        const string SCENE_PATH    = "Assets/Scenes/Main.unity";
        const string APK_REL_PATH  = "../artifacts/FarmQuest.apk";

        [MenuItem("FARM/Build Android APK")]
        public static void BuildAndroid()
        {
            Log("==== TDC build start ====");
            try
            {
#if !ROS2
                // First-pass bootstrap: ROS-TCP-Connector uses #if ROS2 to switch wire
                // format. Add the define + exit cleanly; the wrapper script's second
                // pass will re-enter with ROS2 actually compiled in.
                Log("ROS2 define not present; setting it and exiting for re-run.");
                EnsureROS2Define();
                AssetDatabase.SaveAssets();
                if (Application.isBatchMode) EditorApplication.Exit(0);
                return;
#else
                ConfigurePlayerSettings();
                ConfigureXRForAndroid();
                EnsureMainScene();
                Build();
                Log("==== TDC build complete ====");
                if (Application.isBatchMode) EditorApplication.Exit(0);
#endif
            }
            catch (Exception ex)
            {
                Log("BUILD FAILED: " + ex);
                if (Application.isBatchMode) EditorApplication.Exit(1);
                throw;
            }
        }

        // Bare entrypoint for explicit first-pass invocation.
        public static void EnsureROS2Define()
        {
            var android = UnityEditor.Build.NamedBuildTarget.Android;
            PlayerSettings.GetScriptingDefineSymbols(android, out string[] cur);
            var set = new System.Collections.Generic.HashSet<string>(cur ?? new string[0]);
            if (!set.Contains("ROS2"))
            {
                set.Add("ROS2");
                PlayerSettings.SetScriptingDefineSymbols(android, set.ToArray());
                Log("[settings] added ROS2 to Android scripting defines");
            }
            else
            {
                Log("[settings] ROS2 already in Android scripting defines");
            }
        }

        // ---- Player settings ----

        static void ConfigurePlayerSettings()
        {
            Log("[settings] applying player settings");
            PlayerSettings.companyName = COMPANY_NAME;
            PlayerSettings.productName = PRODUCT_NAME;
            PlayerSettings.SetApplicationIdentifier(
                UnityEditor.Build.NamedBuildTarget.Android, PACKAGE_ID);

            PlayerSettings.Android.minSdkVersion    = AndroidSdkVersions.AndroidApiLevel29;
            PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.forceInternetPermission = true;
            PlayerSettings.Android.useAPKExpansionFiles    = false;
            // debuggable=true lets us adb shell run-as to read logs + tweak PlayerPrefs
            // without rebuilding. Tradeoff: tiny perf hit + a "this app is debuggable"
            // Android toast on first launch. Worth it for the iteration loop.
            EditorUserBuildSettings.development = true;
            EditorUserBuildSettings.allowDebugging = true;

            PlayerSettings.SetScriptingBackend(
                UnityEditor.Build.NamedBuildTarget.Android,
                ScriptingImplementation.IL2CPP);
            PlayerSettings.SetApiCompatibilityLevel(
                UnityEditor.Build.NamedBuildTarget.Android,
                ApiCompatibilityLevel.NET_Standard);

            EnsureROS2Define();

            PlayerSettings.SetGraphicsAPIs(
                BuildTarget.Android,
                new[] { GraphicsDeviceType.OpenGLES3 });
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.gpuSkinning = true;

            // Quest needs no splash screen
            PlayerSettings.SplashScreen.show = false;

            // VR app must be landscape with both eyes
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.LandscapeLeft;
        }

        // ---- XR plugin management + OpenXR/Quest ----

        static void ConfigureXRForAndroid()
        {
            Log("[xr] configuring XR Plugin Management + OpenXR (Meta Quest) for Android");

            // The XR Plugin Management API lives in UnityEditor.XR.Management. We touch
            // it via reflection to avoid hard-coupling this script's compile to the
            // package being present (the package IS in manifest.json, but reflection
            // gives us a clean error path if the meta-openxr resolver fails).
            try
            {
                EnableOpenXRLoaderForAndroid();
                EnableMetaQuestOpenXRFeatures();
            }
            catch (Exception ex)
            {
                Log("[xr] config failed: " + ex.Message);
                throw;
            }
        }

        static void EnableOpenXRLoaderForAndroid()
        {
            var asmMgmtEditor = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Unity.XR.Management.Editor");
            var asmMgmt = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Unity.XR.Management");
            var asmOpenXR = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Unity.XR.OpenXR");
            if (asmMgmtEditor == null || asmMgmt == null || asmOpenXR == null)
                throw new Exception($"XR Management/OpenXR assemblies not loaded "
                    + $"(mgmtEditor={asmMgmtEditor != null}, mgmt={asmMgmt != null}, "
                    + $"openxr={asmOpenXR != null})");

            var loaderType = asmOpenXR.GetType("UnityEngine.XR.OpenXR.OpenXRLoader");
            var perBuildTargetType = asmMgmtEditor.GetType(
                "UnityEditor.XR.Management.XRGeneralSettingsPerBuildTarget");
            var metaStoreType = asmMgmtEditor.GetType(
                "UnityEditor.XR.Management.Metadata.XRPackageMetadataStore");

            // Locate / create per-buildtarget settings asset.
            object settings = perBuildTargetType
                .GetMethod("XRGeneralSettingsForBuildTarget",
                    System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public)
                ?.Invoke(null, new object[] { BuildTargetGroup.Android });

            if (settings == null)
            {
                Log("[xr] no XRGeneralSettings for Android yet; creating");
                // Create the perBuildTarget asset if absent
                var perBT = ScriptableObject.CreateInstance(perBuildTargetType);
                Directory.CreateDirectory("Assets/XR");
                AssetDatabase.CreateAsset(perBT, "Assets/XR/XRGeneralSettingsPerBuildTarget.asset");

                // Add a General Settings object for Android
                var generalSettingsType = asmMgmt.GetType("UnityEngine.XR.Management.XRGeneralSettings");
                var managerSettingsType = asmMgmt.GetType("UnityEngine.XR.Management.XRManagerSettings");

                var general = ScriptableObject.CreateInstance(generalSettingsType);
                general.name = "Android Settings";
                var manager = ScriptableObject.CreateInstance(managerSettingsType);
                manager.name = "Android Manager";
                generalSettingsType.GetProperty("Manager",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance)
                    .SetValue(general, manager);

                AssetDatabase.AddObjectToAsset(general, perBT);
                AssetDatabase.AddObjectToAsset(manager, perBT);

                // perBT.SetSettingsForBuildTarget(BuildTargetGroup.Android, general)
                perBuildTargetType.GetMethod("SetSettingsForBuildTarget",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance)
                    .Invoke(perBT, new object[] { BuildTargetGroup.Android, general });
                AssetDatabase.SaveAssets();
                settings = general;
            }

            // Get the XRManagerSettings (loaders list)
            var manager2 = settings.GetType().GetProperty("Manager").GetValue(settings);

            // Use XRPackageMetadataStore.AssignLoader to wire OpenXRLoader as the loader.
            // signature: bool AssignLoader(XRManagerSettings, string loaderTypeName, BuildTargetGroup)
            var assign = metaStoreType.GetMethod("AssignLoader",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            bool ok = (bool)assign.Invoke(null, new object[] {
                manager2, loaderType.FullName, BuildTargetGroup.Android });
            Log($"[xr] AssignLoader(OpenXRLoader) -> {ok}");

            // Save the EditorBuildSettings link
            // EditorBuildSettings.AddConfigObject("com.unity.xr.management.loader_settings", perBT, true)
            // perBT may not be in EditorBuildSettings yet -- find the perBT asset
            var perBTAsset = AssetDatabase.LoadAssetAtPath(
                "Assets/XR/XRGeneralSettingsPerBuildTarget.asset", perBuildTargetType);
            if (perBTAsset != null)
            {
                EditorBuildSettings.AddConfigObject(
                    "com.unity.xr.management.loader_settings", (UnityEngine.Object)perBTAsset, true);
            }

            AssetDatabase.SaveAssets();
        }

        static void EnableMetaQuestOpenXRFeatures()
        {
            // Use FeatureHelpers (UnityEditor.XR.OpenXR.Features.FeatureHelpers) reflectively to
            // (a) discover + instantiate all OpenXR features for the Android target, then
            // (b) flip `enabled = true` on the Meta Quest support + Oculus Touch features.
            // Without these enabled the Meta Quest manifest hooks no-op, the produced APK
            // lacks the Quest VR markers, and Horizon OS treats it as a 2D app.
            var asmEditor = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Unity.XR.OpenXR.Editor");
            if (asmEditor == null) { Log("[xr] Unity.XR.OpenXR.Editor not loaded"); return; }
            var fh = asmEditor.GetType("UnityEditor.XR.OpenXR.Features.FeatureHelpers");
            if (fh == null) { Log("[xr] FeatureHelpers type missing"); return; }

            // RefreshFeatures discovers all features in project, materializes ScriptableObject
            // instances of any that aren't yet serialized, and adds them to OpenXRSettings.features.
            var refresh = fh.GetMethod("RefreshFeatures",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            refresh.Invoke(null, new object[] { BuildTargetGroup.Android });
            Log("[xr] RefreshFeatures(Android) done");

            string[] featureIds = new[] {
                "com.unity.openxr.feature.metaquest",                       // MetaQuestSupport -- Quest VR manifest hooks
                "com.unity.openxr.feature.input.oculustouch",               // OculusTouchControllerProfile -- controller bindings
                "com.unity.openxr.feature.arfoundation-meta-session",       // ARSession -- required for any AR Foundation feature
                "com.unity.openxr.feature.arfoundation-meta-camera",        // Meta Quest: Camera (Passthrough) -- enables XR_FB_passthrough
            };

            var getFeature = fh.GetMethod("GetFeatureWithIdForBuildTarget",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            foreach (var id in featureIds)
            {
                var feature = (UnityEngine.Object)getFeature.Invoke(null,
                    new object[] { BuildTargetGroup.Android, id });
                if (feature == null) { Log($"[xr] feature not found by id: {id}"); continue; }
                var enabledProp = feature.GetType().GetProperty("enabled",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                if (enabledProp == null) { Log($"[xr] no enabled property on {id}"); continue; }
                enabledProp.SetValue(feature, true);
                EditorUtility.SetDirty(feature);
                Log($"[xr] enabled feature {id}");
            }

            // Passthrough is provided by com.unity.xr.meta-openxr. Feature ids vary by
            // version; enumerate every feature on the Android target's OpenXRSettings and
            // toggle any whose type name mentions "passthrough" so this survives upgrades.
            var asmOpenXR = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Unity.XR.OpenXR");
            var openXRSettingsType = asmOpenXR?.GetType("UnityEngine.XR.OpenXR.OpenXRSettings");
            if (openXRSettingsType != null)
            {
                var getSettings = openXRSettingsType.GetMethod("GetSettingsForBuildTargetGroup",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
                var oxrSettings = getSettings?.Invoke(null, new object[] { BuildTargetGroup.Android });
                var featuresProp = openXRSettingsType.GetProperty("features",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                var allFeatures = featuresProp?.GetValue(oxrSettings) as System.Array;
                if (allFeatures != null)
                {
                    foreach (var f in allFeatures)
                    {
                        if (f == null) continue;
                        var tname = f.GetType().FullName ?? "";
                        if (!tname.ToLower().Contains("passthrough")) continue;
                        var enabledProp = f.GetType().GetProperty("enabled",
                            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                        if (enabledProp == null) continue;
                        enabledProp.SetValue(f, true);
                        EditorUtility.SetDirty((UnityEngine.Object)f);
                        Log($"[xr] enabled passthrough feature: {tname}");
                    }
                }
            }

            AssetDatabase.SaveAssets();
        }

        // ---- Scene ----

        static void EnsureMainScene()
        {
            Directory.CreateDirectory("Assets/Scenes");

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // Main camera at adult-eye height. XR plugin drives its pose from the HMD.
            // CRITICAL for passthrough: backgroundColor alpha=0 so transparent pixels let
            // the Meta Quest compositor show the passthrough layer through.
            var camGO = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener));
            camGO.tag = "MainCamera";
            camGO.transform.position = new Vector3(0f, 1.6f, 0f);
            var cam = camGO.GetComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            // Transparent clear: ARCameraBackground (added below) renders the Meta
            // Quest passthrough feed as a fullscreen background quad behind anything
            // we draw. Camera buffer's alpha=0 pixels show passthrough, opaque pixels
            // (UI) show through normally.
            cam.backgroundColor = new Color(0f, 0f, 0f, 0f);
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane  = 50f;
            cam.allowHDR = false;

            // AR Foundation: enable passthrough rendering on this camera. Components
            // come from com.unity.xr.arfoundation (transitive of meta-openxr).
            AddComponentByName(camGO, "UnityEngine.XR.ARFoundation.ARCameraManager");
            AddComponentByName(camGO, "UnityEngine.XR.ARFoundation.ARCameraBackground");

            // Publisher GO -- reads OpenXR controllers, publishes Q2R topics.
            var pubGO = new GameObject("Q2RPublisher");
            pubGO.AddComponent<Q2RPublisher>();

            // HUD GO -- builds the world-anchored canvas + interactive buttons at runtime.
            var hudGO = new GameObject("HUD");
            hudGO.AddComponent<HUDController>();

            // ARSession (required for ARCameraBackground to drive passthrough).
            var arSessionGO = new GameObject("AR Session");
            AddComponentByName(arSessionGO, "UnityEngine.XR.ARFoundation.ARSession");

            // Controller pointer GOs -- each tracks one OpenXR controller and draws a
            // raycast ray for in-VR button interaction.
            var leftPtrGO = new GameObject("LeftPointer");
            var leftPtr = leftPtrGO.AddComponent<ControllerPointer>();
            leftPtr.IsRight = false;
            var rightPtrGO = new GameObject("RightPointer");
            var rightPtr = rightPtrGO.AddComponent<ControllerPointer>();
            rightPtr.IsRight = true;

            // ROSConnection sentinel GO so the singleton has a place to live in the
            // scene; the Publisher will call GetOrCreateInstance() anyway.
            var rosGO = new GameObject("ROSConnection");
            var rosType = AppDomain.CurrentDomain.GetAssemblies()
                .Select(a => a.GetType("Unity.Robotics.ROSTCPConnector.ROSConnection"))
                .FirstOrDefault(t => t != null);
            if (rosType != null) rosGO.AddComponent(rosType);

            EditorSceneManager.SaveScene(scene, SCENE_PATH);
            Log("[scene] wrote " + SCENE_PATH);
        }

        // ---- Build ----

        static void Build()
        {
            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(SCENE_PATH, true),
            };

            var apkAbs = Path.GetFullPath(Path.Combine(Application.dataPath, "..", APK_REL_PATH));
            Directory.CreateDirectory(Path.GetDirectoryName(apkAbs));

            // Make sure switching to Android is committed.
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
            {
                Log("[build] switching active build target to Android");
                EditorUserBuildSettings.SwitchActiveBuildTarget(
                    BuildTargetGroup.Android, BuildTarget.Android);
            }

            var opts = new BuildPlayerOptions
            {
                scenes = new[] { SCENE_PATH },
                locationPathName = apkAbs,
                target = BuildTarget.Android,
                targetGroup = BuildTargetGroup.Android,
                options = BuildOptions.None,
            };
            Log("[build] target apk: " + apkAbs);
            var report = BuildPipeline.BuildPlayer(opts);
            Log($"[build] result: {report.summary.result}, errors: {report.summary.totalErrors}, "
                + $"warnings: {report.summary.totalWarnings}, size: {report.summary.totalSize}");
            if (report.summary.result != BuildResult.Succeeded)
                throw new Exception($"Build failed: {report.summary.result}");
        }

        static void Log(string s) => Debug.Log("[TDC] " + s);

        // Add a component to a GO by full type name, by reflectively searching all
        // loaded assemblies. Keeps Builder.cs decoupled from packages whose presence
        // we don't want to hard-require at compile time (AR Foundation).
        static UnityEngine.Component AddComponentByName(GameObject go, string typeName)
        {
            var type = AppDomain.CurrentDomain.GetAssemblies()
                .Select(a => a.GetType(typeName))
                .FirstOrDefault(t => t != null);
            if (type == null) { Log($"[scene] component type not found: {typeName}"); return null; }
            var c = go.AddComponent(type);
            Log($"[scene] added {typeName} to {go.name}");
            return c;
        }
    }
}
