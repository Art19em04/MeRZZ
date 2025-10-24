import cv2
from utils.timing import ns_to_ms

def draw_hud(frame, fps, clutch_state, decision, e2e_ms):
    h, w = frame.shape[:2]
    y = 24
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2); y += 24
    cv2.putText(frame, f"Clutch: {clutch_state.name}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2); y += 24
    cv2.putText(frame, f"Gesture: {decision.g.name} conf={decision.confidence:.2f} dur={decision.duration_ms}ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2); y += 24
    cv2.putText(frame, f"e2e p: {e2e_ms:.1f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    return frame


def draw_debug_vectors(frame, p0, p1, x_axis, y_axis, color=(255,255,255)):
    import cv2
    if p0 is None or p1 is None:
        return frame
    h, w = frame.shape[:2]
    def to_px(v):
        return (int(v[0]*w), int(v[1]*h))
    try:
        cv2.arrowedLine(frame, to_px(p0[:2]), to_px(p1[:2]), (0,255,255), 2, tipLength=0.2)
        o = to_px(p0[:2])
        cv2.arrowedLine(frame, o, (o[0]+int(x_axis[0]*80), o[1]+int(x_axis[1]*80)), (255,0,0), 2)
        cv2.arrowedLine(frame, o, (o[0]+int(y_axis[0]*80), o[1]+int(y_axis[1]*80)), (0,0,255), 2)
    except Exception:
        pass
    return frame
