# -*- coding: utf-8 -*-
import time, math
from collections import deque

WRIST=0; THUMB_TIP=4; INDEX_TIP=8; MIDDLE_TIP=12; RING_TIP=16; PINKY_TIP=20
INDEX_PIP=6; MIDDLE_PIP=10; RING_PIP=14; PINKY_PIP=18
INDEX_MCP=5; MIDDLE_MCP=9; RING_MCP=13; PINKY_MCP=17; THUMB_IP=3; THUMB_MCP=2

def _dist(a,b): dx,dy=a[0]-b[0],a[1]-b[1]; return (dx*dx+dy*dy)**0.5
def _angle(a,b,c):
    ab=(a[0]-b[0],a[1]-b[1]); cb=(c[0]-b[0],c[1]-b[1])
    dot=ab[0]*cb[0]+ab[1]*cb[1]; nab=(ab[0]**2+ab[1]**2)**0.5; ncb=(cb[0]**2+cb[1]**2)**0.5
    if nab*ncb==0: return 0.0
    cosv=max(-1.0,min(1.0,dot/(nab*ncb))); return math.acos(cosv)

def finger_flexion(lm):
    def straight(tip,pip,mcp):
        ang=_angle(lm[tip],lm[pip],lm[mcp]); return 1.0-(ang/math.pi)
    return {"index":straight(INDEX_TIP,INDEX_PIP,INDEX_MCP),"middle":straight(MIDDLE_TIP,MIDDLE_PIP,MIDDLE_MCP),
            "ring":straight(RING_TIP,RING_PIP,RING_MCP),"pinky":straight(PINKY_TIP,PINKY_PIP,PINKY_MCP),
            "thumb":straight(THUMB_TIP,THUMB_IP,THUMB_MCP)}

class GestureState:
    def __init__(self,cfg):
        self.cfg=cfg; self.last_emit_global=0.0; self.last_emit_per={}; self.wrist_hist=deque(maxlen=6)
        self.prev_pinch=False; self.hold_latched=False; self.pose_flags={}
    def _can_emit(self,name,now):
        cd=float(self.cfg.get("cooldown_ms",300))
        if now-self.last_emit_global<cd: return False
        need=float(self.cfg.get("per_gesture_min_ms",{}).get(name,cd))
        if now-self.last_emit_per.get(name,0.0)<need: return False
        return True
    def _mark_emit(self,name,now): self.last_emit_global=now; self.last_emit_per[name]=now
    def update_and_classify(self,lm):
        now=time.time()*1000.0
        self.wrist_hist.append((now,lm[WRIST])); swipe=None
        if len(self.wrist_hist)>=2:
            t0,p0=self.wrist_hist[0]; t1,p1=self.wrist_hist[-1]; dt=max(1e-3,(t1-t0)/1000.0)
            vx=(p1[0]-p0[0])/dt
            if abs(vx) > 1.0*(self.cfg.get("swipe_speed_px",800)/800.0):
                swipe = "SWIPE_RIGHT" if vx>0 else "SWIPE_LEFT"
        pinch_d=_dist(lm[THUMB_TIP],lm[INDEX_TIP]); pinch=pinch_d<self.cfg.get("pinch_threshold",0.045)
        flex=finger_flexion(lm); avg_other=(flex["middle"]+flex["ring"]+flex["pinky"])/3.0
        tu_thumb=self.cfg.get("thumbs_up_thumb_max_flex",0.35); tu_others=self.cfg.get("thumbs_up_others_min_flex",0.5)
        pt_idx=self.cfg.get("point_index_max_flex",0.30); pt_others=self.cfg.get("point_others_min_flex",0.5)
        fist_thr=self.cfg.get("fist_threshold",0.35)
        is_open=(flex["index"]<0.35 and flex["middle"]<0.35 and flex["ring"]<0.35 and flex["pinky"]<0.35)
        is_fist=(flex["index"]>fist_thr and flex["middle"]>fist_thr and flex["ring"]>fist_thr and flex["pinky"]>fist_thr)
        is_thumbs_up=(flex["thumb"]<tu_thumb and avg_other>tu_others)
        is_point=(flex["index"]<pt_idx and avg_other>pt_others)
        self.pose_flags={"OPEN_PALM":bool(is_open),"FIST":bool(is_fist),"THUMBS_UP":bool(is_thumbs_up),
                         "POINT":bool(is_point),"PINCH":bool(pinch),
                         "SWIPE_LEFT": swipe=="SWIPE_LEFT", "SWIPE_RIGHT": swipe=="SWIPE_RIGHT"}
        clutch=self.cfg.get("clutch","none"); ready=True if clutch=="none" else pinch
        emit=None
        if swipe and ready and self._can_emit(swipe,now): emit=swipe
        elif not self.prev_pinch and pinch and ready and self._can_emit("PINCH_TAP",now): emit="PINCH_TAP"
        elif pinch and ready:
            if not self.hold_latched and self._can_emit("PINCH_HOLD",now): emit="PINCH_HOLD"; self.hold_latched=True
        else: self.hold_latched=False
        if is_fist and ready and self._can_emit("FIST",now): emit=emit or "FIST"
        if is_thumbs_up and ready and self._can_emit("THUMBS_UP",now): emit=emit or "THUMBS_UP"
        if is_point and ready and self._can_emit("POINT",now): emit=emit or "POINT"
        if is_open and ready and self._can_emit("OPEN_PALM",now): emit=emit or "OPEN_PALM"
        if emit: self._mark_emit(emit,now)
        self.prev_pinch=pinch; return emit
