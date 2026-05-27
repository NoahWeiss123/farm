// Intentionally a no-op. We used to draw a small sphere at each
// controller's tracked pose for in-headset feedback, but with
// passthrough engaged the user already sees their actual hands holding
// the controllers — the synthetic dots showed up at the bottom of the
// FOV (the controllers' physical Y is below the user's gaze) and were
// just confusing extra geometry.
//
// Kept as a stub so existing scene references (RightPointer /
// LeftPointer GameObjects in Main.unity) still resolve and don't emit
// MissingScript warnings.

using UnityEngine;

namespace TeleopDataCollector
{
    public class ControllerPointer : MonoBehaviour
    {
        public bool IsRight = true;
    }
}
