import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from moviepy.editor import ImageSequenceClip

# -------------------------------------------------------------------
# 1. 최종 분석 엔진 (모든 관절 떨림 보정 기능 탑재)
# -------------------------------------------------------------------
def analyze_swing(video_path, progress_bar):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 진행률 표시를 위한 전체 프레임 수

    # --- 스무딩 및 분석 변수 설정 ---
    output_frames = []
    window_size = 7  # 스무딩 창 크기
    landmark_buffers = {i: [] for i in range(33)} # 33개 모든 관절을 위한 버퍼

    swing_plane_line = None
    min_y_coord = 9999
    top_position_coords = None
    address_wrist_y = 0
    backswing_height = 0
    impact_hand_y = 0
    top_detected = False
    right_hand_path = []

    if not cap.isOpened():
        return None, None, None

    display_height = 720
    display_width = int(display_height * 9 / 16)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # 진행률 업데이트
        progress_bar.progress(frame_count / total_frames)

        resized_frame = cv2.resize(frame, (display_width, display_height))
        image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.pose_landmarks:
            # 모든 관절에 이동 평균 필터 적용
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                buffer = landmark_buffers[i]
                buffer.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                if len(buffer) > window_size:
                    buffer.pop(0)

                smoothed_point = np.mean(buffer, axis=0)
                # 기존 landmark 값을 스무딩된 값으로 덮어쓰기
                landmark.x, landmark.y, landmark.z, landmark.visibility = smoothed_point

            # 이제 'results.pose_landmarks' 자체가 보정된 값을 가짐
            landmarks = results.pose_landmarks.landmark
            right_wrist_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            if right_wrist_landmark.visibility > 0.7:
                if swing_plane_line is None:
                    # 어드레스 자세 설정
                    shoulder_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
                    shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
                    wrist_x = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x + right_wrist_landmark.x) / 2
                    wrist_y = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y + right_wrist_landmark.y) / 2
                    p1 = (int(wrist_x * display_width), int(wrist_y * display_height))
                    p2 = (int(shoulder_x * display_width), int(shoulder_y * display_height))
                    swing_plane_line = (p1, p2)
                    address_wrist_y = p1[1]
                
                right_wrist_coord = (int(right_wrist_landmark.x * display_width), int(right_wrist_landmark.y * display_height))
                right_hand_path.append(right_wrist_coord)

                # 백스윙 탑 위치 계산
                if right_wrist_coord[1] < min_y_coord:
                    min_y_coord = right_wrist_coord[1]
                    top_position_coords = right_wrist_coord
                    top_detected = True
                    if address_wrist_y > 0:
                        backswing_height = address_wrist_y - min_y_coord
                
                # 임팩트 순간 손 높이 계산
                if top_detected and impact_hand_y == 0 and right_wrist_coord[1] >= address_wrist_y * 0.95:
                    impact_hand_y = right_wrist_coord[1]

        # --- 화면에 그리기 ---
        text = f"Backswing Height: {backswing_height} px"
        cv2.putText(resized_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if swing_plane_line:
            cv2.line(resized_frame, swing_plane_line[0], swing_plane_line[1], color=(255, 0, 0), thickness=2)
        if len(right_hand_path) > 1:
            cv2.polylines(resized_frame, [np.array(right_hand_path)], isClosed=False, color=(0, 255, 0), thickness=3)
        if top_position_coords:
            cv2.circle(resized_frame, top_position_coords, 10, (0, 255, 255), 3)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image=resized_frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

        output_frames.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # --- 분석 결과 및 그래프 생성 ---
    analysis_results = {
        "max_height": backswing_height,
        "impact_height_vs_address": impact_hand_y - address_wrist_y if impact_hand_y > 0 else 0
    }

    fig = None
    if right_hand_path:
        hand_heights = [p[1] for p in right_hand_path]
        hand_heights_inverted = [display_height - h for h in hand_heights]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hand_heights_inverted, color='green', label='Hand Height')
        ax.set_title('Hand Height Over Time')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Hand Height (pixels from bottom)')
        ax.legend()
        ax.grid(True)

    output_video_path = None
    if output_frames:
        output_video_path = os.path.join(tempfile.gettempdir(), 'swing_output_final.mp4')
        clip = ImageSequenceClip(output_frames, fps=fps)
        clip.write_videofile(output_video_path, codec='libx264', logger=None)

    return output_video_path, fig, analysis_results

# -------------------------------------------------------------------
# 2. 스트림릿(Streamlit) 웹사이트 UI 코드
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="AI 골프 스윙 분석")
st.title("🏌️ AI 골프 스윙 자동 분석 리포트")
st.info("개선된 AI 엔진이 탑재되어 모든 관절의 떨림을 보정하고 더 정확하게 분석합니다.")

uploaded_file = st.file_uploader("MP4 형식의 스윙 영상을 업로드하세요.", type="mp4")

if uploaded_file is not None:
    # 임시 파일로 저장하여 경로 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    st.header("✅ 업로드된 원본 영상")
    st.video(temp_video_path)

    if st.button("분석 시작!"):
        # 진행률 표시줄 생성
        progress_bar = st.progress(0, text="스윙을 분석 중입니다... 잠시만 기다려 주세요.")
        
        # 개선된 분석 함수 호출
        result_video_path, result_graph, insights = analyze_swing(temp_video_path, progress_bar)
        
        # 분석 완료 후 진행률 100%로 채우고 메시지 표시
        progress_bar.progress(1.0, text="분석 완료!")

        if result_video_path:
            st.success("분석이 성공적으로 완료되었습니다!")

            col1, col2 = st.columns(2)
            with col1:
                st.header("분석 결과 영상")
                with open(result_video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)

            with col2:
                st.header("손 높이 그래프")
                st.pyplot(result_graph)

            st.header("‍⚕️ 자동 스윙 진단 리포트")
            st.metric("최대 백스윙 높이 (어드레스 대비)", f"{insights['max_height']} px")
            impact_metric = insights['impact_height_vs_address']
            st.metric("임팩트 시 손 높이 (어드레스 대비)", f"{impact_metric} px")

            if impact_metric <= 10:
                st.info("✔️ **임팩트 포지션**: 좋습니다! 어드레스와 유사한 좋은 임팩트 높이를 유지했습니다.")
            else:
                st.warning("⚠️ **개선점**: 임팩트 시 손이 어드레스보다 높게 뜨는 경향이 있습니다. 상체가 일찍 일어나는 '배치기' 동작일 수 있으니 확인해 보세요.")
        
        else:
            st.error("영상 분석에 실패했습니다. 다른 영상을 시도해 보거나, 영상이 너무 짧지 않은지 확인해 주세요.")
