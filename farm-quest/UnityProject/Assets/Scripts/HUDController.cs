// Minimal, fixed-in-space FARM status card.
//
// Layout: one panel, ~30 cm wide, parked 1.2 m in front of the user at
// 1.45 m height on the first valid head pose. Never re-parks — the
// card stays put in the world as the user moves around (which is what
// the user explicitly asked for).
//
// Information surfaced (in order, top to bottom):
//   • "FARM" wordmark + the laptop IP / link state         (subtle)
//   • Big primary line: "● REC  00:42" while recording,
//     "READY" while idle, "CONTROLLING" while the trigger drives
//     the arm                                              (state colour)
//   • Footer help: A start/save · B cancel · trigger move · grip gripper

using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;

namespace TeleopDataCollector
{
    public class HUDController : MonoBehaviour
    {
        Transform hudRoot;
        Text statusLine;
        Text primaryLine;
        Text footerLine;
        bool placed;

        // Compatibility shim — Q2RPublisher and friends used to push
        // controller-input snapshots into this static for the HUD to
        // read. Kept as a no-op signature so callers compile if any
        // still reference it.
        public static int PulseButtonUpperFrames;
        public static void RecordInputs(XRNode _, RosMessageTypes.Quest2ros.OVR2ROSInputsMsg __) { }

        // Recording state mirrored from the laptop daemon. The
        // Q2RPublisher polls /v1/recording/state at low rate and
        // sets this so the HUD can render the live timer.
        public static bool RecordingActive;
        public static float RecordingElapsedS;
        public static int RecordingFrameCount;

        // ── palette ──────────────────────────────────────────────────────
        static readonly Color C_BG        = new Color(0.08f, 0.08f, 0.10f, 0.92f);
        static readonly Color C_BORDER    = new Color(0.30f, 0.32f, 0.40f, 0.55f);
        static readonly Color C_DIM       = new Color(0.62f, 0.66f, 0.72f, 1f);
        static readonly Color C_FG        = new Color(0.96f, 0.97f, 0.98f, 1f);
        static readonly Color C_OK        = new Color(0.40f, 0.85f, 0.55f, 1f);
        static readonly Color C_AMBER     = new Color(0.96f, 0.74f, 0.32f, 1f);
        static readonly Color C_REC       = new Color(0.95f, 0.32f, 0.34f, 1f);
        static readonly Color C_CTRL      = new Color(0.40f, 0.78f, 0.96f, 1f);

        void Awake()
        {
            ConnectionConfig.LoadFromPrefs();
            BuildCard();
            Application.targetFrameRate = 90;
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

            // 540 × 250 UI pixels → ~0.38 m × 0.175 m at 0.0007 scale.
            var rt = (RectTransform)canvasGO.transform;
            rt.sizeDelta = new Vector2(540, 250);
            canvasGO.transform.localScale = Vector3.one * 0.0007f;

            // Background fill
            var bg = NewChild(canvasGO, "card_bg");
            var bgImg = bg.AddComponent<Image>();
            bgImg.color = C_BG;
            FillParent(bg);

            // Subtle border (a slightly larger background tinted lighter,
            // offset behind the main fill)
            var border = NewChild(canvasGO, "card_border");
            var brt = (RectTransform)border.transform;
            brt.anchorMin = Vector2.zero; brt.anchorMax = Vector2.one;
            brt.offsetMin = new Vector2(-2, -2); brt.offsetMax = new Vector2(2, 2);
            var brdImg = border.AddComponent<Image>();
            brdImg.color = C_BORDER;
            border.transform.SetSiblingIndex(0);

            // Top status line (small, dim) — "FARM • <ip> · <link>"
            statusLine = NewText(canvasGO, "status",
                new Vector2(0, 86), 22, TextAnchor.MiddleCenter, C_DIM);
            statusLine.text = "FARM · connecting…";

            // Primary state line (large) — REC timer / READY / CONTROLLING
            primaryLine = NewText(canvasGO, "primary",
                new Vector2(0, 6), 56, TextAnchor.MiddleCenter, C_FG);
            primaryLine.fontStyle = FontStyle.Bold;
            primaryLine.text = "READY";

            // Footer help (very dim, two-line key map)
            footerLine = NewText(canvasGO, "footer",
                new Vector2(0, -82), 18, TextAnchor.MiddleCenter, C_DIM);
            footerLine.text = "A start/save   B cancel\ntrigger move    grip gripper";
        }

        void Update()
        {
            // Place once on first valid head pose. We attach the card
            // as a child of the camera's parent (the XR tracking rig),
            // not the camera itself — that way it stays put in the
            // world while still moving with the rig when OpenXR
            // recenters (e.g. after the user takes the headset off /
            // back on). Without this the card visibly drifts every
            // time tracking re-centers.
            if (!placed)
            {
                var cam = Camera.main;
                if (cam != null && cam.transform.position.sqrMagnitude > 0.0001f)
                {
                    var rigParent = cam.transform.parent;
                    if (rigParent != null) hudRoot.SetParent(rigParent, false);
                    var fwd = cam.transform.forward; fwd.y = 0f;
                    if (fwd.sqrMagnitude < 0.01f) fwd = Vector3.forward;
                    fwd.Normalize();
                    // Place in *rig-local* space so we live in the tracking
                    // origin's frame, not world frame. Use a fixed point ~1.2 m
                    // ahead of the user's first-look direction, chest height.
                    var origin = new Vector3(cam.transform.position.x, 0f, cam.transform.position.z);
                    hudRoot.position = origin + fwd * 1.2f + Vector3.up * 1.45f;
                    hudRoot.rotation = Quaternion.LookRotation(fwd);
                    placed = true;
                }
            }

            // Status line: link state + endpoint IP.
            var pub = Object.FindFirstObjectByType<Q2RPublisher>();
            if (pub != null && pub.IsLinked)
            {
                statusLine.text = $"FARM   linked  {ConnectionConfig.Ip}";
                statusLine.color = C_OK;
            }
            else
            {
                statusLine.text = $"FARM   {ConnectionConfig.Ip}:{ConnectionConfig.Port}";
                statusLine.color = C_AMBER;
            }

            // Primary line: REC > CONTROLLING > READY
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
            }
            else if (triggerHeld)
            {
                primaryLine.text = "CONTROLLING";
                primaryLine.color = C_CTRL;
            }
            else
            {
                primaryLine.text = "READY";
                primaryLine.color = C_FG;
            }
        }

        // ── tiny helpers ──────────────────────────────────────────────────
        static GameObject NewChild(GameObject parent, string name)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent.transform, false);
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
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent.transform, false);
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
