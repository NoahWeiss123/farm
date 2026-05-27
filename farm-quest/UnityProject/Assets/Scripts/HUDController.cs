// FARM floating-tablet HUD.
//
// Placement: parks once on first valid head pose, ~1.2 m in front of the
// user at ~1.45 m height. Position then stays fixed in physical space.
// Rotation billboards toward the head every frame so the panel always
// reads at the same angle no matter where the user walks.
//
// What it shows (polled from /v1/hud at ~2 Hz, no camera frames):
//   • FARM ip + link state
//   • Mode chip: DIGITAL / DIGITAL + REAL
//   • CAMERAS row with one green/red dot per camera
//   • EPISODES count
//   • Big primary line: ● REC mm:ss / READY / CONTROLLING
//   • ZERO and E-STOP buttons (clickable via LeftHandUIPointer)
//   • A red border lights up around the card when recording
//
// LeftHandUIPointer hit-tests against `Buttons` and invokes the action
// when the left trigger is pulled.

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using UnityEngine.XR;

namespace TeleopDataCollector
{
    public class HUDController : MonoBehaviour
    {
        public struct UIButton
        {
            public RectTransform rect;
            public Action onClick;
            public string label;
        }

        public Transform CardTransform => cardTransform;
        public Vector2 CardSizePx => cardSizePx;
        public IReadOnlyList<UIButton> Buttons => buttons;

        Transform hudRoot;
        Transform cardTransform;
        Vector2 cardSizePx = new Vector2(600f, 380f);

        Text statusLine;
        Text modeChip;
        Image modeChipBg;
        Text camLabel;
        readonly List<Image> camDots = new List<Image>();
        readonly List<Text>  camTexts = new List<Text>();
        Text episodeLabel;
        Text episodeCount;
        Text primaryLine;
        Image recBorder;     // outer red glow that turns on while recording
        Text footerLine;

        Button zeroBtn;
        Button estopBtn;
        readonly List<UIButton> buttons = new List<UIButton>();

        bool placed;

        // Mirrored from the bridge (Q2RPublisher edge-detects A/B). The HUD
        // poll also overwrites these with daemon truth so a daemon-side
        // recording stays in sync if it was started elsewhere.
        public static bool RecordingActive;
        public static float RecordingElapsedS;
        public static int RecordingFrameCount;

        // Compatibility shim left over from the legacy HUD — kept so older
        // code paths still compile without referencing it.
        public static int PulseButtonUpperFrames;
        public static void RecordInputs(XRNode _, RosMessageTypes.Quest2ros.OVR2ROSInputsMsg __) { }

        // Latest /v1/hud snapshot (filled by the poller coroutine).
        struct Cam { public string name; public bool alive; }
        readonly List<Cam> hudCameras = new List<Cam>();
        int hudEpisodes = 0;
        bool driveRealArm = false;
        bool daemonReachable = false;

        // ── palette ──────────────────────────────────────────────────────
        static readonly Color C_BG        = new Color(0.07f, 0.08f, 0.10f, 0.95f);
        static readonly Color C_BORDER    = new Color(0.30f, 0.34f, 0.42f, 0.85f);
        static readonly Color C_DIM       = new Color(0.62f, 0.66f, 0.72f, 1f);
        static readonly Color C_FG        = new Color(0.96f, 0.97f, 0.98f, 1f);
        static readonly Color C_OK        = new Color(0.40f, 0.85f, 0.55f, 1f);
        static readonly Color C_BAD       = new Color(0.90f, 0.32f, 0.32f, 1f);
        static readonly Color C_AMBER     = new Color(0.96f, 0.74f, 0.32f, 1f);
        static readonly Color C_REC       = new Color(0.95f, 0.32f, 0.34f, 1f);
        static readonly Color C_CTRL      = new Color(0.40f, 0.78f, 0.96f, 1f);
        static readonly Color C_BTN       = new Color(0.18f, 0.20f, 0.24f, 1f);
        static readonly Color C_BTN_HOVER = new Color(0.26f, 0.30f, 0.36f, 1f);
        static readonly Color C_BTN_DANGER= new Color(0.40f, 0.14f, 0.16f, 1f);
        static readonly Color C_BTN_DANGER_HOVER = new Color(0.62f, 0.18f, 0.20f, 1f);
        static readonly Color C_REC_GLOW  = new Color(0.95f, 0.22f, 0.24f, 1f);
        static readonly Color C_REC_GLOW_OFF = new Color(0, 0, 0, 0);
        static readonly Color C_MODE_DIGITAL = new Color(0.20f, 0.28f, 0.40f, 1f);
        static readonly Color C_MODE_REAL    = new Color(0.55f, 0.20f, 0.22f, 1f);

        void Awake()
        {
            ConnectionConfig.LoadFromPrefs();
            BuildCard();
            Application.targetFrameRate = 90;
            StartCoroutine(PollHudLoop());
        }

        void BuildCard()
        {
            var root = new GameObject("FARM_HUD_Root");
            hudRoot = root.transform;

            var canvasGO = new GameObject("FARM_HUD_Canvas");
            canvasGO.transform.SetParent(hudRoot, false);
            var canvas = canvasGO.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasGO.AddComponent<CanvasScaler>();
            canvasGO.AddComponent<GraphicRaycaster>();

            cardTransform = canvasGO.transform;
            var rt = (RectTransform)canvasGO.transform;
            rt.sizeDelta = cardSizePx;
            canvasGO.transform.localScale = Vector3.one * 0.0007f;

            // Recording glow — outer rectangle behind everything else.
            // Width/height extends past the card; alpha keyed by RecordingActive.
            var glow = NewChild(canvasGO, "rec_glow");
            var glowRT = (RectTransform)glow.transform;
            glowRT.anchorMin = Vector2.zero; glowRT.anchorMax = Vector2.one;
            glowRT.offsetMin = new Vector2(-22, -22);
            glowRT.offsetMax = new Vector2( 22,  22);
            recBorder = glow.AddComponent<Image>();
            recBorder.color = C_REC_GLOW_OFF;

            // Subtle persistent border
            var border = NewChild(canvasGO, "card_border");
            var brt = (RectTransform)border.transform;
            brt.anchorMin = Vector2.zero; brt.anchorMax = Vector2.one;
            brt.offsetMin = new Vector2(-4, -4);
            brt.offsetMax = new Vector2( 4,  4);
            border.AddComponent<Image>().color = C_BORDER;

            // Background fill
            var bg = NewChild(canvasGO, "card_bg");
            FillParent(bg);
            bg.AddComponent<Image>().color = C_BG;

            // ── header row ────────────────────────────────────────────────
            statusLine = NewText(canvasGO, "status",
                new Vector2(-160, 168), 20, TextAnchor.MiddleLeft, C_DIM);
            statusLine.text = "FARM · connecting…";
            ((RectTransform)statusLine.transform).sizeDelta = new Vector2(260, 32);

            // Mode chip — small pill in top-right corner
            var chip = NewChild(canvasGO, "mode_chip");
            var chipRT = (RectTransform)chip.transform;
            chipRT.sizeDelta = new Vector2(220, 36);
            chipRT.anchoredPosition = new Vector2(160, 168);
            modeChipBg = chip.AddComponent<Image>();
            modeChipBg.color = C_MODE_DIGITAL;
            modeChip = NewText(chip, "mode_text", Vector2.zero, 20, TextAnchor.MiddleCenter, C_FG);
            modeChip.fontStyle = FontStyle.Bold;
            ((RectTransform)modeChip.transform).sizeDelta = new Vector2(220, 36);
            modeChip.text = "DIGITAL";

            // ── cameras row ───────────────────────────────────────────────
            camLabel = NewText(canvasGO, "cam_label",
                new Vector2(-242, 108), 18, TextAnchor.MiddleLeft, C_DIM);
            camLabel.text = "CAMERAS";
            ((RectTransform)camLabel.transform).sizeDelta = new Vector2(120, 28);

            // Two slots reserved for base + wrist; the poller may populate
            // names dynamically but we lay out the dots up front.
            for (int i = 0; i < 2; i++)
            {
                var slot = NewChild(canvasGO, $"cam_slot_{i}");
                var slotRT = (RectTransform)slot.transform;
                slotRT.sizeDelta = new Vector2(180, 32);
                slotRT.anchoredPosition = new Vector2(-30 + i * 160, 108);

                var dot = NewChild(slot, "dot");
                var dotRT = (RectTransform)dot.transform;
                dotRT.sizeDelta = new Vector2(18, 18);
                dotRT.anchoredPosition = new Vector2(-72, 0);
                var dotImg = dot.AddComponent<Image>();
                dotImg.color = C_BAD;
                camDots.Add(dotImg);

                var txt = NewText(slot, "name", new Vector2(8, 0), 22, TextAnchor.MiddleLeft, C_FG);
                ((RectTransform)txt.transform).sizeDelta = new Vector2(140, 32);
                txt.text = i == 0 ? "base" : "wrist";
                camTexts.Add(txt);
            }

            // ── episodes row ──────────────────────────────────────────────
            episodeLabel = NewText(canvasGO, "ep_label",
                new Vector2(-242, 60), 18, TextAnchor.MiddleLeft, C_DIM);
            episodeLabel.text = "EPISODES";
            ((RectTransform)episodeLabel.transform).sizeDelta = new Vector2(120, 28);

            episodeCount = NewText(canvasGO, "ep_count",
                new Vector2(-100, 60), 26, TextAnchor.MiddleLeft, C_FG);
            episodeCount.fontStyle = FontStyle.Bold;
            ((RectTransform)episodeCount.transform).sizeDelta = new Vector2(120, 36);
            episodeCount.text = "0";

            // ── primary state line ────────────────────────────────────────
            primaryLine = NewText(canvasGO, "primary",
                new Vector2(0, -10), 54, TextAnchor.MiddleCenter, C_FG);
            primaryLine.fontStyle = FontStyle.Bold;
            ((RectTransform)primaryLine.transform).sizeDelta = new Vector2(560, 80);
            primaryLine.text = "READY";

            // ── action buttons ────────────────────────────────────────────
            zeroBtn = BuildButton(canvasGO, "btn_zero",
                new Vector2(-130, -100), new Vector2(220, 76),
                "ZERO", C_BTN, C_BTN_HOVER, OnZeroPressed);
            estopBtn = BuildButton(canvasGO, "btn_estop",
                new Vector2( 130, -100), new Vector2(220, 76),
                "E-STOP", C_BTN_DANGER, C_BTN_DANGER_HOVER, OnEstopPressed);

            // ── footer ────────────────────────────────────────────────────
            footerLine = NewText(canvasGO, "footer",
                new Vector2(0, -168), 16, TextAnchor.MiddleCenter, C_DIM);
            footerLine.text = "A start/save · B cancel · right trigger move · grip gripper";
            ((RectTransform)footerLine.transform).sizeDelta = new Vector2(560, 22);
        }

        Button BuildButton(GameObject canvasGO, string name, Vector2 pos, Vector2 size,
                           string label, Color baseCol, Color hoverCol, Action onClick)
        {
            var go = NewChild(canvasGO, name);
            var rt = (RectTransform)go.transform;
            rt.sizeDelta = size;
            rt.anchoredPosition = pos;

            var img = go.AddComponent<Image>();
            img.color = baseCol;
            var btn = go.AddComponent<Button>();
            btn.targetGraphic = img;
            var colors = btn.colors;
            colors.normalColor = baseCol;
            colors.highlightedColor = hoverCol;
            colors.pressedColor = hoverCol;
            colors.selectedColor = baseCol;
            btn.colors = colors;
            btn.onClick.AddListener(() => onClick?.Invoke());

            var txt = NewText(go, "label", Vector2.zero, 30, TextAnchor.MiddleCenter, C_FG);
            txt.fontStyle = FontStyle.Bold;
            ((RectTransform)txt.transform).sizeDelta = size;
            txt.text = label;

            buttons.Add(new UIButton { rect = rt, onClick = onClick, label = label });
            return btn;
        }

        void Update()
        {
            // Place once on first valid head pose, then billboard each frame.
            var cam = Camera.main;
            if (cam == null) return;

            if (!placed)
            {
                if (cam.transform.position.sqrMagnitude > 0.0001f)
                {
                    var rigParent = cam.transform.parent;
                    if (rigParent != null) hudRoot.SetParent(rigParent, false);
                    var fwd = cam.transform.forward; fwd.y = 0f;
                    if (fwd.sqrMagnitude < 0.01f) fwd = Vector3.forward;
                    fwd.Normalize();
                    var origin = new Vector3(cam.transform.position.x, 0f, cam.transform.position.z);
                    hudRoot.position = origin + fwd * 1.2f + Vector3.up * 1.45f;
                    hudRoot.rotation = Quaternion.LookRotation(fwd, Vector3.up);
                    placed = true;
                }
            }

            if (placed)
            {
                // Billboard: yaw-only so the panel stays vertical and
                // readable even if the user looks up/down. Smooth toward
                // target rotation to avoid jitter on small head motions.
                var fromCam = hudRoot.position - cam.transform.position;
                fromCam.y = 0f;
                if (fromCam.sqrMagnitude > 0.0001f)
                {
                    var target = Quaternion.LookRotation(fromCam.normalized, Vector3.up);
                    hudRoot.rotation = Quaternion.Slerp(
                        hudRoot.rotation, target, Time.deltaTime * 8f);
                }
            }

            // Status line — link state + endpoint IP
            var pub = UnityEngine.Object.FindFirstObjectByType<Q2RPublisher>();
            if (pub != null && pub.IsLinked)
            {
                statusLine.text = $"FARM   linked   {ConnectionConfig.Ip}";
                statusLine.color = C_OK;
            }
            else
            {
                statusLine.text = $"FARM   {ConnectionConfig.Ip}:{ConnectionConfig.Port}";
                statusLine.color = C_AMBER;
            }

            // Mode chip
            if (driveRealArm)
            {
                modeChip.text = "DIGITAL + REAL";
                modeChipBg.color = C_MODE_REAL;
            }
            else
            {
                modeChip.text = "DIGITAL";
                modeChipBg.color = C_MODE_DIGITAL;
            }

            // Camera dots
            for (int i = 0; i < camDots.Count; i++)
            {
                if (i < hudCameras.Count)
                {
                    camTexts[i].text = hudCameras[i].name;
                    camDots[i].color = hudCameras[i].alive ? C_OK : C_BAD;
                    camTexts[i].color = hudCameras[i].alive ? C_FG : C_DIM;
                }
                else
                {
                    camTexts[i].text = i == 0 ? "base" : "wrist";
                    camDots[i].color = daemonReachable ? C_BAD : new Color(0.4f, 0.4f, 0.4f, 1f);
                    camTexts[i].color = C_DIM;
                }
            }

            episodeCount.text = hudEpisodes.ToString();

            // Primary state line: REC > CONTROLLING > READY
            var right = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
            float trig = 0f;
            if (right.isValid) right.TryGetFeatureValue(CommonUsages.trigger, out trig);
            bool triggerHeld = trig > 0.5f;

            if (RecordingActive)
            {
                int sec = Mathf.FloorToInt(RecordingElapsedS);
                int mm = sec / 60, ss = sec % 60;
                primaryLine.text = $"● REC   {mm:00}:{ss:00}";
                primaryLine.color = C_REC;
                recBorder.color = C_REC_GLOW;
            }
            else if (triggerHeld)
            {
                primaryLine.text = "CONTROLLING";
                primaryLine.color = C_CTRL;
                recBorder.color = C_REC_GLOW_OFF;
            }
            else
            {
                primaryLine.text = "READY";
                primaryLine.color = C_FG;
                recBorder.color = C_REC_GLOW_OFF;
            }
        }

        // ── HTTP poll ─────────────────────────────────────────────────────
        IEnumerator PollHudLoop()
        {
            while (true)
            {
                string url = $"http://{ConnectionConfig.Ip}:8787/v1/hud";
                using (var req = UnityWebRequest.Get(url))
                {
                    req.timeout = 2;
                    yield return req.SendWebRequest();
                    if (req.result == UnityWebRequest.Result.Success)
                    {
                        daemonReachable = true;
                        ApplyHudPayload(req.downloadHandler.text);
                    }
                    else
                    {
                        daemonReachable = false;
                    }
                }
                yield return new WaitForSeconds(0.5f);
            }
        }

        // JsonUtility can't parse dict-shaped JSON; the daemon returns
        // cameras as a flat array of {name, alive} so we can deserialize
        // it without writing a hand-rolled parser.
        [Serializable] class HudCam { public string name; public bool alive; }
        [Serializable] class HudRec { public bool recording; public float elapsed_s; public int frame_count; }
        [Serializable] class HudPayload {
            public HudCam[] cameras;
            public int episodes;
            public HudRec recording;
            public bool drive_real_arm;
        }

        void ApplyHudPayload(string json)
        {
            HudPayload p;
            try { p = JsonUtility.FromJson<HudPayload>(json); }
            catch { return; }
            if (p == null) return;

            hudCameras.Clear();
            if (p.cameras != null)
                foreach (var c in p.cameras)
                    hudCameras.Add(new Cam { name = c.name, alive = c.alive });

            hudEpisodes = p.episodes;
            driveRealArm = p.drive_real_arm;

            // Daemon truth wins for recording state, but only if it
            // disagrees — A/B edges still drive local timing between polls.
            if (p.recording != null)
            {
                RecordingActive = p.recording.recording;
                if (RecordingActive)
                {
                    RecordingElapsedS = p.recording.elapsed_s;
                    RecordingFrameCount = p.recording.frame_count;
                }
            }
        }

        // ── button actions ────────────────────────────────────────────────
        void OnZeroPressed()
        {
            StartCoroutine(PostNoBody($"http://{ConnectionConfig.Ip}:8787/v1/teleop/home"));
        }

        void OnEstopPressed()
        {
            StartCoroutine(PostNoBody($"http://{ConnectionConfig.Ip}:8787/v1/teleop/estop"));
        }

        IEnumerator PostNoBody(string url)
        {
            using (var req = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST))
            {
                req.uploadHandler = new UploadHandlerRaw(new byte[0]);
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");
                req.timeout = 2;
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success)
                    Debug.LogWarning($"[FARM HUD] POST {url} failed: {req.error}");
            }
        }

        // ── tiny helpers ──────────────────────────────────────────────────
        static GameObject NewChild(GameObject parent, string name)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent.transform, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot     = new Vector2(0.5f, 0.5f);
            return go;
        }
        static void FillParent(GameObject go)
        {
            var rt = (RectTransform)go.transform;
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
        }
        static Text NewText(GameObject parent, string name, Vector2 pos, int size,
                            TextAnchor align, Color color)
        {
            var go = NewChild(parent, name);
            var t = go.AddComponent<Text>();
            t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            t.fontSize = size;
            t.alignment = align;
            t.color = color;
            t.horizontalOverflow = HorizontalWrapMode.Overflow;
            t.verticalOverflow = VerticalWrapMode.Overflow;
            var rt = (RectTransform)go.transform;
            rt.sizeDelta = new Vector2(500, 70);
            rt.anchoredPosition = pos;
            return t;
        }
    }
}
