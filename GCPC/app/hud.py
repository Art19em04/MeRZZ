import cv2


def draw_hud(frame, fps, clutch_state, decision, e2e_ms):
    y = 24
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 24
    cv2.putText(frame, f"Clutch: {clutch_state.name}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y += 24
    cv2.putText(
        frame,
        f"Gesture: {decision.g.name} conf={decision.confidence:.2f} dur={decision.duration_ms}ms",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2,
    )
    y += 24
    cv2.putText(frame, f"e2e p: {e2e_ms:.1f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return frame
