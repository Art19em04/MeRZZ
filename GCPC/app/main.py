# -*- coding: utf-8 -*-
import os, json, time, cv2
from PySide6 import QtWidgets
from app.gestures import GestureState
from app.os_events_win import press_combo, mouse_move_normalized, mouse_press, mouse_release
from app.osd import OSD
from app.tracker_mediapipe import MediaPipeHandTracker

APP_DIR = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(APP_DIR)

def load_config():
    with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as f: return json.load(f)

class DebouncedTrigger:
    def __init__(self, dwell_ms=260, refractory_ms=900):
        self.dwell_ms=dwell_ms; self.refractory_ms=refractory_ms; self.candidate_since=None; self.last_fire=0
    def update(self, now_ms, active):
        if not active: self.candidate_since=None; return False
        if self.candidate_since is None: self.candidate_since=now_ms; return False
        if (now_ms-self.candidate_since)>=self.dwell_ms and (now_ms-self.last_fire)>=self.refractory_ms:
            self.last_fire=now_ms; self.candidate_since=None; return True
        return False

def open_camera(idx, w, h):
    def try_open(i, api):
        cap = cv2.VideoCapture(i, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, _ = cap.read()
            if ok: return cap
            cap.release()
        return None
    for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = try_open(idx, api)
        if cap:
            print(f"[videoio] open idx={idx} api={api}")
            return cap
    for probe in range(0,6):
        for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = try_open(probe, api)
            if cap:
                print(f"[videoio] open idx={probe} api={api}")
                return cap
    return None

def draw_landmarks(frame, lm):
    h,w=frame.shape[:2]
    for (x,y) in lm: 
        cv2.circle(frame,(int(x*w),int(y*h)),3,(0,255,0),-1)

def main():
    cfg=load_config(); app=QtWidgets.QApplication([]); osd=OSD(); osd.show()
    vcfg=cfg["video"]; idx=int(vcfg.get("camera_index",0)); w=int(vcfg.get("width",1280)); h=int(vcfg.get("height",720)); mirror=bool(vcfg.get("mirror",True)); show_fps=bool(vcfg.get("show_fps",False))
    cap=open_camera(idx,w,h); 
    if cap is None: raise RuntimeError("Не удалось открыть камеру")

    dcfg=cfg.get("detector",{})
    tracker = MediaPipeHandTracker(min_det=dcfg.get("min_detection_confidence",0.6),
                                   min_trk=dcfg.get("min_tracking_confidence",0.5),
                                   max_hands=dcfg.get("max_num_hands",2),
                                   model_complexity=dcfg.get("model_complexity",1))

    gR=GestureState(cfg["gesture_engine"]); gL=GestureState(cfg["gesture_engine"])

    seq_cfg=cfg.get("sequence",{}); bcfg=cfg.get("bimanual",{})
    arm_delay_ms=int(seq_cfg.get("arm_delay_ms",420)); refractory_ms=int(seq_cfg.get("refractory_ms",1100))
    cancel_exit_ms=int(seq_cfg.get("cancel_on_hand_exit_ms",900)); auto_exit=bool(seq_cfg.get("auto_exit_on_hand_exit",True))
    max_len=int(seq_cfg.get("max_len",6)); exit_on_commit=bool(seq_cfg.get("exit_on_commit",True))

    confirm_deb=DebouncedTrigger(int(bcfg.get("confirm_dwell_ms",220)), int(bcfg.get("confirm_refractory_ms",700)))
    commit_deb =DebouncedTrigger(int(bcfg.get("commit_dwell_ms",260)), int(bcfg.get("commit_refractory_ms",1200)))

    cmd_map=cfg.get("command_mappings",{})
    single_map=cmd_map.get("single_gestures",{})
    seq_map={}
    seq_map.update(cmd_map.get("complex_gestures",{}))
    seq_map.update(single_map)
    if not seq_map:
        seq_map=cfg.get("sequence_mappings",{})

    dominant_hand=str(cfg.get("dominant_hand","right")).lower()
    one_cfg=cfg.get("one_hand_mode",{})
    one_enabled=bool(one_cfg.get("enabled",True))
    raw_hold_pose=str(one_cfg.get("hold_pose","FIST")).upper()
    one_hold_pose=raw_hold_pose
    if raw_hold_pose.startswith("LEFT_") or raw_hold_pose.startswith("RIGHT_"):
        one_hold_pose=raw_hold_pose.split("_",1)[1]
    one_hold_display=one_cfg.get("hold_display") or raw_hold_pose
    one_status_label=one_cfg.get("status_label") or "ONE-HAND"
    non_dom_label="LEFT" if dominant_hand!="left" else "RIGHT"
    default_hint=f"{non_dom_label} HOLD: {one_hold_display}"
    one_active_hint=one_cfg.get("active_hint") or default_hint
    one_block_sequences=bool(one_cfg.get("block_sequences",True))

    mouse_cfg=cfg.get("mouse_control",{})
    mouse_enabled=bool(mouse_cfg.get("enabled",True))
    raw_mouse_pose=str(mouse_cfg.get("activation_pose","LEFT_THUMBS_UP")).upper()
    mouse_hold_pose=raw_mouse_pose
    if raw_mouse_pose.startswith("LEFT_") or raw_mouse_pose.startswith("RIGHT_"):
        mouse_hold_pose=raw_mouse_pose.split("_",1)[1]
    mouse_status_label=mouse_cfg.get("status_label") or "MOUSE"
    mouse_active_hint=mouse_cfg.get("active_hint") or f"{non_dom_label} HOLD: {mouse_hold_pose.replace('_',' ')}"
    mouse_smooth=max(0.0,min(1.0,float(mouse_cfg.get("smoothing_alpha",0.25))))
    pointer_hand=str(mouse_cfg.get("pointer_hand","right")).lower()

    mouse_prev=None
    mouse_left_down=False
    mouse_right_down=False

    seq_active=False; seq_buffer=[]; last_evt_ms=0; last_sent_ms=0
    last_seen_R=int(time.time()*1000); last_seen_L=int(time.time()*1000)
    pending_R=None; last_R_event_ms=0
    last_R_label=""; last_L_label=""
    left_open_ts=None; undo_window_ms=int(bcfg.get("undo_window_ms",900))
    one_hand_active=False; mouse_active=False; last_single_action=""

    fps=None; last_frame_time=time.time()

    while True:
        QtWidgets.QApplication.processEvents()
        ret, frame = cap.read()
        if not ret: break
        if mirror: frame=cv2.flip(frame,1)

        now=time.time(); dt=now-last_frame_time; last_frame_time=now
        now_ms=int(now*1000)
        if show_fps and dt>0:
            inst=1.0/dt
            fps = inst if fps is None else (0.9*fps + 0.1*inst)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        hands=tracker.process(rgb)  # 0..2 hands

        # Split by label with fallback by x-position
        right=left=None; rights=[]; lefts=[]
        for hnd in hands:
            if hnd.get("label","")=="Right": rights.append(hnd)
            elif hnd.get("label","")=="Left": lefts.append(hnd)
        if not rights and not lefts and hands:
            # fallback: sort by x of wrist (0) — левую/правую определим по положению
            srt=sorted(hands, key=lambda h: h["lm"][0][0])  # по x WRIST
            if len(srt)==2: lefts=[srt[0]]; rights=[srt[1]]
            elif len(srt)==1:
                # классифицируем эвристикой (x thumb vs pinky)
                lm=srt[0]["lm"]; lbl="Right" if lm[4][0] > lm[20][0] else "Left"
                if lbl=="Right": rights=[srt[0]]
                else: lefts=[srt[0]]

        right = max(rights, key=lambda h: h.get("score",0)) if rights else None
        left  = max(lefts,  key=lambda h: h.get("score",0)) if lefts else None

        # Update gesture states
        evR=None; evL=None

        if right:
            last_seen_R=now_ms; evR=gR.update_and_classify(right["lm"]) or ""
            if evR: last_R_label=evR
        else:
            gR.pose_flags.clear()
            if seq_active and cancel_exit_ms>0 and (now_ms-last_seen_R)>=cancel_exit_ms:
                seq_buffer.clear(); pending_R=None
                if auto_exit: seq_active=False
                print("[SEQ] Авто-отмена: правая рука вне кадра")

        if left:
            last_seen_L=now_ms; evL=gL.update_and_classify(left["lm"]) or ""
            if evL: last_L_label=evL
        else:
            gL.pose_flags.clear()

        dominant_is_right = dominant_hand != "left"
        dom_event = evR if dominant_is_right else evL
        if not dom_event: dom_event=None
        dom_present = bool(right) if dominant_is_right else bool(left)
        non_dom_present = bool(left) if dominant_is_right else bool(right)
        non_dom_state = gL if dominant_is_right else gR
        hold_flag=False
        if one_enabled and non_dom_present:
            hold_flag = non_dom_state.pose_flags.get(one_hold_pose, False)
            if not hold_flag and one_hold_pose=="PINCH":
                hold_flag = non_dom_state.pose_flags.get("PINCH", False)
        one_hand_active = bool(hold_flag) if one_enabled else False
        if not one_hand_active:
            last_single_action=""

        mouse_flag=False
        if mouse_enabled and non_dom_present:
            mouse_flag = non_dom_state.pose_flags.get(mouse_hold_pose, False)
            if not mouse_flag and mouse_hold_pose=="PINCH":
                mouse_flag = non_dom_state.pose_flags.get("PINCH", False)
        mouse_active = bool(mouse_flag) if mouse_enabled else False
        if mouse_active:
            one_hand_active=False
        if not mouse_active:
            mouse_prev=None
            if mouse_left_down:
                mouse_release("left"); mouse_left_down=False
            if mouse_right_down:
                mouse_release("right"); mouse_right_down=False

        # Start by two palms
        both_open = (gR.pose_flags.get("OPEN_PALM",False) and gL.pose_flags.get("OPEN_PALM",False))
        if not seq_active and both_open and not (one_block_sequences and one_hand_active) and not mouse_active:
            seq_active=True; seq_buffer.clear(); pending_R=None; last_evt_ms=now_ms
            print("[SEQ] Режим ввода: СТАРТ")

        if one_hand_active and not seq_active and dom_event and dom_present and not mouse_active:
            combo=single_map.get(dom_event)
            if combo and (now_ms-last_sent_ms)>=refractory_ms:
                print(f"[ONE-HAND] {dom_event} -> {combo}"); press_combo(combo); last_sent_ms=now_ms
                last_single_action=f"LAST: {dom_event} → {combo}"

        if mouse_active:
            pointer_source = right if pointer_hand=="right" else left
            pointer_state = gR if pointer_hand=="right" else gL
            if pointer_source:
                tip=pointer_source["lm"][8]
                target=(tip[0], tip[1])
                if mouse_prev is None or mouse_smooth<=0.0:
                    mouse_prev=target
                else:
                    prev_x, prev_y = mouse_prev
                    mouse_prev=(prev_x + (target[0]-prev_x)*(1.0-mouse_smooth),
                                 prev_y + (target[1]-prev_y)*(1.0-mouse_smooth))
                mx,my=mouse_prev
                mouse_move_normalized(mx,my)
                is_fist=pointer_state.pose_flags.get("FIST",False)
                if is_fist and not mouse_left_down:
                    mouse_press("left"); mouse_left_down=True
                if not is_fist and mouse_left_down:
                    mouse_release("left"); mouse_left_down=False
                is_open=pointer_state.pose_flags.get("OPEN_PALM",False)
                if is_open and not mouse_right_down:
                    mouse_press("right"); mouse_right_down=True
                if not is_open and mouse_right_down:
                    mouse_release("right"); mouse_right_down=False
            else:
                if mouse_left_down:
                    mouse_release("left"); mouse_left_down=False
                if mouse_right_down:
                    mouse_release("right"); mouse_right_down=False

        # Right: candidate (ignore control poses)
        if seq_active and right:
            ev = last_R_label
            if ev and ev not in ("OPEN_PALM","FIST"):
                if (pending_R is None) or (pending_R!=ev and (now_ms-last_R_event_ms)>=arm_delay_ms):
                    pending_R=ev; last_R_event_ms=now_ms

        # Left: confirm with tap
        left_tap = (last_L_label=="PINCH_TAP")
        if seq_active and pending_R and confirm_deb.update(now_ms, left_tap):
            if len(seq_buffer)<max_len and (now_ms-last_evt_ms)>=arm_delay_ms:
                seq_buffer.append(pending_R); print(f"[SEQ] +{pending_R}  buffer={seq_buffer}")
                pending_R=None; last_evt_ms=now_ms

        # Left: undo = OPEN -> FIST
        if seq_active and left:
            if gL.pose_flags.get("OPEN_PALM",False):
                left_open_ts = now_ms
            if gL.pose_flags.get("FIST",False) and left_open_ts and (now_ms-left_open_ts)<=undo_window_ms:
                if seq_buffer:
                    popped=seq_buffer.pop(); print(f"[SEQ] UNDO -{popped}  buffer={seq_buffer}")
                left_open_ts=None

        # Commit: two fists
        both_fists = (gR.pose_flags.get("FIST",False) and gL.pose_flags.get("FIST",False))
        if seq_active and commit_deb.update(now_ms, both_fists):
            key="+".join(seq_buffer) if seq_buffer else None
            combo=seq_map.get(key) if key else None
            if combo and (now_ms-last_sent_ms)>=refractory_ms:
                print(f"[SEQ-COMMIT] {key} -> {combo}"); press_combo(combo); last_sent_ms=now_ms
            else:
                print(f"[SEQ-COMMIT] Нет маппинга для: {key}")
            seq_buffer.clear(); pending_R=None; last_evt_ms=now_ms
            if exit_on_commit: seq_active=False

        # OSD text
        if seq_active:
            top = "REC"
            sub = ("BUF: "+ "+".join(seq_buffer[-6:])) if seq_buffer else "BUF: —"
            if pending_R: sub += f"   |   CAND: {pending_R}"
            if mouse_active and mouse_enabled:
                sub += f"   |   {mouse_status_label}"
            if one_hand_active and one_enabled:
                sub += f"   |   {one_status_label}"
        else:
            if mouse_active and mouse_enabled:
                top = mouse_status_label
                sub = mouse_active_hint
            elif one_hand_active and one_enabled:
                top = one_status_label
                sub = one_active_hint
                if last_single_action:
                    sub += f"   |   {last_single_action}"
            else:
                top = "IDLE"
                sub = "BUF: —"
        osd.set_text(top, sub)

        # debug draw
        if right: draw_landmarks(frame, right["lm"])
        if left:  draw_landmarks(frame, left["lm"])
        if show_fps and fps is not None:
            cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("GCPC - Camera", frame)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
