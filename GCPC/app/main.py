# -*- coding: utf-8 -*-
import os, json, time, cv2
from PySide6 import QtWidgets
from app.gestures import GestureState
from app.os_events_win import press_combo
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
    vcfg=cfg["video"]; idx=int(vcfg.get("camera_index",0)); w=int(vcfg.get("width",1280)); h=int(vcfg.get("height",720)); mirror=bool(vcfg.get("mirror",True))
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

    seq_map=cfg.get("sequence_mappings",{})

    seq_active=False; seq_buffer=[]; last_evt_ms=0; last_sent_ms=0
    last_seen_R=int(time.time()*1000); last_seen_L=int(time.time()*1000)
    pending_R=None; last_R_event_ms=0
    last_R_label=""; last_L_label=""
    left_open_ts=None; undo_window_ms=int(bcfg.get("undo_window_ms",900))

    while True:
        QtWidgets.QApplication.processEvents()
        ret, frame = cap.read()
        if not ret: break
        if mirror: frame=cv2.flip(frame,1)

        now_ms=int(time.time()*1000)
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

        # Start by two palms
        both_open = (gR.pose_flags.get("OPEN_PALM",False) and gL.pose_flags.get("OPEN_PALM",False))
        if not seq_active and both_open:
            seq_active=True; seq_buffer.clear(); pending_R=None; last_evt_ms=now_ms
            print("[SEQ] Режим ввода: СТАРТ")

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
        top = "REC" if seq_active else "IDLE"
        sub = ("BUF: "+ "+".join(seq_buffer[-6:])) if seq_buffer else "BUF: —"
        if pending_R: sub += f"   |   CAND: {pending_R}"
        osd.set_text(top, sub)

        # debug draw
        if right: draw_landmarks(frame, right["lm"])
        if left:  draw_landmarks(frame, left["lm"])
        cv2.imshow("GCPC - Camera", frame)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
