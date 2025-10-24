import cv2, time, os, json, math
from cv_io import Video
from hands import HandTracker
from clutch import PinchClutch, ClutchState
from microgestures import MicroGestureDetector, G
from os_events import send_keys
from telemetry import Telemetry
from hud import draw_hud
from config import load_config
from utils.timing import now_ns, ns_to_ms, Tick

KEYMAP = {
    G.SWIPE_UP: "swipe_up",
    G.SWIPE_DOWN: "swipe_down",
    G.TAP_FORWARD: "tap_forward",
    G.TAP_BACK: "tap_back",
}

def main():
    cfg = load_config()
    tel = Telemetry(cfg)
    vid = Video(cfg)
    tracker = HandTracker(cfg)
    clutch = PinchClutch(cfg)
    mgest = MicroGestureDetector(cfg)

    hud_on = cfg["hud"].get("enabled", True)
    fps_acc = 0.0
    fps_alpha = 0.1
    last_ns = now_ns()
    last_action = None
    last_action_ts = 0

    os.makedirs(cfg["telemetry"]["dir"], exist_ok=True)

    while True:
        frame = vid.read()
        if frame is None:
            break
        t = now_ns()
        dt = t - last_ns
        last_ns = t
        inst_fps = 1.0 / (dt / 1e9) if dt else 0.0
        fps_acc = fps_alpha * inst_fps + (1 - fps_alpha) * fps_acc

        tick = Tick(t_frame_in=t)
        timestamp_ms = int(time.monotonic() * 1000)

        res = tracker.process(frame, timestamp_ms)
        if res:
            tick.t_landmarks = now_ns()
            cstate = clutch.update(res.landmarks, timestamp_ms)
            decision = mgest.update(res.landmarks, timestamp_ms, cstate==ClutchState.WINDOW)
        else:
            cstate = ClutchState.IDLE
            decision = mgest.update(None, timestamp_ms, False)

        fired = False
        if decision.g != G.NONE and decision.confidence >= cfg["gesture"]["conf_min"]:
            tick.t_decision = now_ns()
            key_name = KEYMAP[decision.g]
            keys = cfg["mapping"][key_name]
            send_keys(keys)
            tick.t_os_event = now_ns()
            fired = True
            last_action = decision.g.name
            last_action_ts = timestamp_ms

        # Telemetry
        tel.write({
            "t_frame_in_ns": tick.t_frame_in,
            "t_landmarks_ns": tick.t_landmarks,
            "t_decision_ns": tick.t_decision,
            "t_os_event_ns": tick.t_os_event,
            "fps": fps_acc,
            "gesture": decision.g.name,
            "confidence": decision.confidence,
            "duration_ms": decision.duration_ms,
            "clutch_state": cstate.name,
            "action_sent": fired,
            "env": {
                "platform": os.name
            }
        })

        # HUD
        if hud_on:
            # draw ROI
            x,y,w,h = cfg["video"]["roi"]
            H,W = frame.shape[:2]
            cv2.rectangle(frame,(int(x*W),int(y*H)), (int((x+w)*W),int((y+h)*H)), (0,255,0),2)
            e2e = (tick.t_os_event - tick.t_frame_in)/1e6 if tick.t_os_event else 0.0
            frame = draw_hud(frame, fps_acc, cstate, decision, e2e)
        cv2.imshow("microgestures_mvp", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # space
            hud_on = not hud_on
        elif key == 0x77:  # 'w' toggles ROI box thickness (placeholder, can be used to debug)
            pass
        elif key == 0x70:  # F1 placeholder
            pass
        # F8 marking false positive (on Windows it's VK_F8; in OpenCV this may differ by backend; use ASCII '8' + SHIFT as fallback)
        elif key == 0x7f:  # DEL (fallback)
            pass
        # try detect F8 specifically via getWindowProperty (not always possible); we provide Alt+F as alternative
        # We allow user to hit 'f' to mark FP
        elif key == ord('f'):
            tel.write({"mark": "false_positive", "ts_ms": int(time.time()*1000), "last_action": last_action})

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
