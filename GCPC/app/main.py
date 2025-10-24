import os, sys, time, cv2
try:
    # локальные импорты (если запускаешь "python app/main.py")
    from cv_io import Video
    from hands import HandTracker
    from clutch import PinchClutch, ClutchState
    from microgestures import MicroGestureDetector, G
    from os_events import send_keys
    from telemetry import Telemetry
    from hud import draw_hud
    from config import load_config
    from utils.timing import now_ns, Tick
    from factory import get_hand_tracker
except Exception:
    # запуск как модуль: "python -m app.main"
    from app.cv_io import Video
    from app.hands import HandTracker
    from app.clutch import PinchClutch, ClutchState
    from app.microgestures import MicroGestureDetector, G
    from app.os_events import send_keys
    from app.telemetry import Telemetry
    from app.hud import draw_hud
    from app.config import load_config
    from app.utils.timing import now_ns, Tick
    from app.factory import get_hand_tracker

# --- консоль без буферизации (важно для PyCharm/Windows) ---
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

KEYMAP = {
    G.SWIPE_UP: "swipe_up",
    G.SWIPE_DOWN: "swipe_down",
    G.TAP_FORWARD: "tap_forward",
    G.TAP_BACK: "tap_back",
}

def reasons_not_firing(decision, cstate, gcfg, last_action_ts, now_ms):
    rs = []
    if cstate != ClutchState.WINDOW:
        rs.append("no_window")
    if decision.confidence < gcfg["conf_min"]:
        rs.append("low_conf")
    dur = int(decision.duration_ms or 0)
    if not (gcfg["min_duration_ms_eff"] <= dur <= gcfg["max_duration_ms_eff"]):
        rs.append(f"bad_duration({dur} not in {gcfg['min_duration_ms_eff']}..{gcfg['max_duration_ms_eff']})")
    if (now_ms - last_action_ts) < gcfg["cooldown_ms"]:
        rs.append("cooldown")
    return rs


def main():
    cfg = load_config()
    tel = Telemetry(cfg)
    vid = Video(cfg)
    tracker = get_hand_tracker(cfg)
    print(f"[INIT] tracker={cfg.get('tracker', 'mediapipe')}", flush=True)
    clutch = PinchClutch(cfg)
    mgest = MicroGestureDetector(cfg)

    hud_on = cfg["hud"].get("enabled", True)

    fps_acc = 0.0
    fps_alpha = 0.1
    last_ns = now_ns()
    last_action = None
    last_action_ts = 0

    prev_cstate = ClutchState.IDLE
    last_hb_ns = last_ns
    last_hand_log_ms = 0

    os.makedirs(cfg["telemetry"]["dir"], exist_ok=True)

    while True:
        frame = vid.read()
        if frame is None:
            print("[ERR] camera frame is None; stopping.", flush=True)
            break

        t_ns = now_ns()
        dt_ns = t_ns - last_ns
        last_ns = t_ns
        inst_fps = (1.0 / (dt_ns / 1e9)) if dt_ns > 0 else 0.0
        fps_acc = fps_alpha * inst_fps + (1 - fps_alpha) * fps_acc

        gcfg = cfg["gesture"]
        fps_eff = max(10.0, min(60.0, fps_acc if fps_acc > 1e-3 else 30.0))
        frames_min = 3
        frames_max = 9
        gcfg["min_duration_ms_eff"] = max(gcfg.get("min_duration_ms", 60),
                                          int(round(1000.0 * frames_min / fps_eff)) - 40)
        gcfg["max_duration_ms_eff"] = min(gcfg.get("max_duration_ms", 600),
                                          int(round(1000.0 * frames_max / fps_eff)) + 80)


        tick = Tick(t_frame_in=t_ns)
        timestamp_ms = int(time.monotonic() * 1000)

        res = tracker.process(frame, timestamp_ms)
        if not res or (res.score < cfg["tracker_opts"].get("min_det_conf", 0.5)):
            cstate = ClutchState.IDLE
            decision = mgest.update(None, timestamp_ms, False)
        else:
            tick.t_landmarks = now_ns()
            cstate = clutch.update(res.landmarks, timestamp_ms)
            decision = mgest.update(res.landmarks, timestamp_ms, cstate == ClutchState.WINDOW)

        # лог переходов клатча
        if cstate != prev_cstate:
            print(f"[CLUTCH] {prev_cstate.name} → {cstate.name}", flush=True)
            prev_cstate = cstate

        # раз в 300мс — факт наличия руки/handedness
        if (timestamp_ms - last_hand_log_ms) >= 300:
            if res:
                print(f"[HANDS] {res.handedness} score={res.score:.2f}", flush=True)
            else:
                print("[HANDS] no hand", flush=True)
            last_hand_log_ms = timestamp_ms

        # кандидат в жест — печатаем всегда
        if decision.g != G.NONE:
            print(f"[CAND] g={decision.g.name} conf={decision.confidence:.2f} "
                  f"dur={decision.duration_ms}ms thr={cfg['gesture']['conf_min']}", flush=True)

        # принятие решения
        fired = False
        ok_conf = decision.confidence >= gcfg["conf_min"]
        ok_dur = gcfg["min_duration_ms_eff"] <= decision.duration_ms <= gcfg["max_duration_ms_eff"]
        ok_win = (cstate == ClutchState.WINDOW)

        if decision.g != G.NONE and ok_conf and ok_dur and ok_win:
            tick.t_decision = now_ns()
            key_name = KEYMAP[decision.g]
            keys = cfg["mapping"][key_name]
            send_keys(keys)
            tick.t_os_event = now_ns()
            fired = True
            last_action = decision.g.name
            last_action_ts = timestamp_ms
            print(f"[FIRE] {key_name} → {keys}", flush=True)
        else:
            if decision.g != G.NONE:
                rs = reasons_not_firing(decision, cstate, gcfg, last_action_ts, timestamp_ms)
                if rs:
                    print("[DROP] " + ",".join(rs), flush=True)

        # телеметрия
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
            "env": {"platform": os.name}
        })

        # HUD
        if hud_on:
            x, y, w, h = cfg["video"]["roi"]
            H, W = frame.shape[:2]
            cv2.rectangle(frame, (int(x * W), int(y * H)),
                          (int((x + w) * W), int((y + h) * H)), (0, 255, 0), 2)
            e2e = (tick.t_os_event - tick.t_frame_in) / 1e6 if tick.t_os_event else 0.0
            frame = draw_hud(frame, fps_acc, cstate, decision, e2e)
        cv2.imshow("microgestures_mvp", frame)

        # heartbeat раз в секунду, чтобы видеть «жив ли цикл»
        if (t_ns - last_hb_ns) >= 1_000_000_000:
            print(f"[HB] fps={fps_acc:.1f} clutch={cstate.name} "
                  f"g={decision.g.name} conf={decision.confidence:.2f}", flush=True)
            last_hb_ns = t_ns

        # клавиши окна
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            hud_on = not hud_on
        elif key == ord('f'):
            # пометка FP
            print("[FP] mark false positive", flush=True)
            from time import time as _now
            tel.write({"mark": "false_positive", "ts_ms": int(_now() * 1000), "last_action": last_action})

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
