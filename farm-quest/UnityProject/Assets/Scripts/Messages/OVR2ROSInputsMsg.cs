// Hand-written C# of quest2ros/OVR2ROSInputs. Matches the wire format used by
// the upstream Quest2ROS app + the host quest_teleop_node subscriber. Pattern
// mirrors the generated *Msg classes in com.unity.robotics.ros-tcp-connector.

using System;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.Quest2ros
{
    [Serializable]
    public class OVR2ROSInputsMsg : Message
    {
        public const string k_RosMessageName = "quest2ros/OVR2ROSInputs";
        public override string RosMessageName => k_RosMessageName;

        public bool button_upper;
        public bool button_lower;
        public float thumb_stick_horizontal;
        public float thumb_stick_vertical;
        public float press_index;
        public float press_middle;

        public OVR2ROSInputsMsg() { }

        public OVR2ROSInputsMsg(bool button_upper, bool button_lower,
                                float thumb_stick_horizontal, float thumb_stick_vertical,
                                float press_index, float press_middle)
        {
            this.button_upper = button_upper;
            this.button_lower = button_lower;
            this.thumb_stick_horizontal = thumb_stick_horizontal;
            this.thumb_stick_vertical = thumb_stick_vertical;
            this.press_index = press_index;
            this.press_middle = press_middle;
        }

        public static OVR2ROSInputsMsg Deserialize(MessageDeserializer d) => new OVR2ROSInputsMsg(d);

        OVR2ROSInputsMsg(MessageDeserializer d)
        {
            d.Read(out this.button_upper);
            d.Read(out this.button_lower);
            d.Read(out this.thumb_stick_horizontal);
            d.Read(out this.thumb_stick_vertical);
            d.Read(out this.press_index);
            d.Read(out this.press_middle);
        }

        public override void SerializeTo(MessageSerializer s)
        {
            s.Write(this.button_upper);
            s.Write(this.button_lower);
            s.Write(this.thumb_stick_horizontal);
            s.Write(this.thumb_stick_vertical);
            s.Write(this.press_index);
            s.Write(this.press_middle);
        }

        public override string ToString() =>
            $"OVR2ROSInputs(U={button_upper} L={button_lower} stick=({thumb_stick_horizontal:F2},{thumb_stick_vertical:F2}) trig={press_index:F2} grip={press_middle:F2})";

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
#else
        [UnityEngine.RuntimeInitializeOnLoadMethod]
#endif
        public static void Register() => MessageRegistry.Register(k_RosMessageName, Deserialize);
    }
}
