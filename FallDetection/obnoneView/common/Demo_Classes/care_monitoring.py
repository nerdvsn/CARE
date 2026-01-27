# Care Monitoring Demo for Elderly/Patient Care
# Features: Fall Detection, Vital Signs, Bed Monitoring, Activity Tracking, Alerts
#
# Author: Custom Implementation for IWR6843-ODS
# Based on Texas Instruments mmWave SDK

from collections import deque
import numpy as np
import time
import json
import os
from datetime import datetime

from PySide2.QtCore import Qt, QThread, QTimer, Signal, QObject
from PySide2.QtGui import QFont, QColor
from PySide2.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QComboBox, QCheckBox, QSlider, QFormLayout, QPushButton,
    QSpinBox, QDoubleSpinBox, QTextEdit, QFrame, QScrollArea, QLineEdit
)
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from Common_Tabs.plot_3d import Plot3D
from Common_Tabs.plot_1d import Plot1D
from Demo_Classes.Helper_Classes.fall_detection import FallDetection, FallDetectionSliderClass
from demo_defines import DEVICE_DEMO_DICT
from graph_utilities import get_trackColors, eulerRot
from gl_text import GLTextItem
from gui_threads import updateQTTargetThread3D

import logging
log = logging.getLogger(__name__)


# ============================================================================
# ALERT SYSTEM - For notifications and logging
# ============================================================================
class AlertSystem(QObject):
    """Handles alerts and notifications for care events"""
    alertTriggered = Signal(str, str, str)  # type, message, timestamp

    def __init__(self):
        super().__init__()
        self.alerts = []
        self.alert_callbacks = []
        self.log_file = None
        self.enable_logging = True

    def setup_logging(self, log_dir="care_logs"):
        """Setup logging to file"""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"care_log_{timestamp}.json")

    def trigger_alert(self, alert_type, message, severity="INFO"):
        """Trigger an alert"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": timestamp
        }
        self.alerts.append(alert)
        self.alertTriggered.emit(alert_type, message, timestamp)

        # Log to file
        if self.enable_logging and self.log_file:
            self._write_log(alert)

        # Execute callbacks
        for callback in self.alert_callbacks:
            callback(alert)

        log.info(f"[{severity}] {alert_type}: {message}")

    def _write_log(self, alert):
        """Write alert to log file"""
        try:
            logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            logs.append(alert)
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            log.error(f"Failed to write log: {e}")

    def register_callback(self, callback):
        """Register a callback for alerts"""
        self.alert_callbacks.append(callback)

    def get_recent_alerts(self, count=10):
        """Get most recent alerts"""
        return self.alerts[-count:] if self.alerts else []


# ============================================================================
# ZONE DEFINITIONS - For bed, room, bathroom detection
# ============================================================================
class Zone:
    """Defines a monitoring zone (bed, bathroom, room boundary)"""
    def __init__(self, name, x_range, y_range, z_range=None):
        self.name = name
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range if z_range else (-10, 10)

    def contains(self, x, y, z=0):
        """Check if point is inside zone"""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)

    def to_dict(self):
        return {
            "name": self.name,
            "x_range": (self.x_min, self.x_max),
            "y_range": (self.y_min, self.y_max),
            "z_range": (self.z_min, self.z_max)
        }


# ============================================================================
# ACTIVITY TRACKER - Monitors movement and activity levels
# ============================================================================
class ActivityTracker:
    """Tracks activity levels and patterns"""

    def __init__(self, history_seconds=300, frame_time_ms=55):
        self.frame_time = frame_time_ms
        self.history_len = int((history_seconds * 1000) / frame_time_ms)

        # Activity metrics
        self.velocity_history = deque(maxlen=self.history_len)
        self.position_history = deque(maxlen=self.history_len)
        self.activity_scores = deque(maxlen=self.history_len)

        # Statistics
        self.total_distance = 0
        self.last_position = None
        self.activity_level = "Unknown"  # Low, Medium, High

    def update(self, tracks):
        """Update activity metrics with new track data"""
        if len(tracks) == 0:
            self.velocity_history.append(0)
            self.activity_scores.append(0)
            return

        # Use first tracked person
        track = tracks[0]
        x, y, z = track[1], track[2], track[3]
        vx, vy, vz = track[4], track[5], track[6]

        # Calculate velocity magnitude
        velocity = np.sqrt(vx**2 + vy**2 + vz**2)
        self.velocity_history.append(velocity)

        # Track distance moved
        if self.last_position is not None:
            dist = np.sqrt((x - self.last_position[0])**2 +
                          (y - self.last_position[1])**2)
            self.total_distance += dist
        self.last_position = (x, y, z)
        self.position_history.append((x, y, z))

        # Calculate activity score (0-100)
        score = min(100, velocity * 50)  # Scale velocity to score
        self.activity_scores.append(score)

        # Determine activity level
        avg_score = np.mean(list(self.activity_scores)[-100:]) if len(self.activity_scores) > 0 else 0
        if avg_score < 10:
            self.activity_level = "Low"
        elif avg_score < 40:
            self.activity_level = "Medium"
        else:
            self.activity_level = "High"

    def get_activity_score(self):
        """Get current activity score (0-100)"""
        if len(self.activity_scores) == 0:
            return 0
        return np.mean(list(self.activity_scores)[-50:])

    def get_statistics(self):
        """Get activity statistics"""
        return {
            "activity_level": self.activity_level,
            "activity_score": self.get_activity_score(),
            "total_distance_m": round(self.total_distance, 2),
            "avg_velocity": np.mean(list(self.velocity_history)) if self.velocity_history else 0
        }


# ============================================================================
# BED MONITOR - Tracks bed occupancy and sleep patterns
# ============================================================================
class BedMonitor:
    """Monitors bed occupancy, time in bed, and sleep patterns"""

    # States
    STATE_UNKNOWN = "Unknown"
    STATE_IN_BED = "In Bed"
    STATE_OUT_OF_BED = "Out of Bed"
    STATE_GETTING_UP = "Getting Up"
    STATE_LYING_DOWN = "Lying Down"

    def __init__(self, bed_zone, frame_time_ms=55):
        self.bed_zone = bed_zone
        self.frame_time = frame_time_ms

        # State tracking
        self.current_state = self.STATE_UNKNOWN
        self.previous_state = self.STATE_UNKNOWN
        self.state_start_time = time.time()

        # Time tracking
        self.time_in_bed_today = 0  # seconds
        self.time_out_of_bed = 0
        self.last_update_time = time.time()

        # Out of bed detection
        self.out_of_bed_threshold_sec = 60  # Alert if out of bed > 60 sec at night
        self.is_night_mode = False

        # Statistics
        self.bed_entries = 0
        self.bed_exits = 0
        self.bathroom_visits_count = 0

        # Height tracking for lying vs sitting
        self.height_threshold = 0.6  # meters - below this = lying down

    def update(self, tracks, height_data=None):
        """Update bed monitor with new track data"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        in_bed_zone = False
        current_height = None

        if len(tracks) > 0:
            track = tracks[0]
            x, y, z = track[1], track[2], track[3]
            in_bed_zone = self.bed_zone.contains(x, y)
            current_height = z

        # State machine
        old_state = self.current_state

        if in_bed_zone:
            if current_height is not None and current_height < self.height_threshold:
                self.current_state = self.STATE_LYING_DOWN
            else:
                self.current_state = self.STATE_IN_BED
            self.time_in_bed_today += dt
            self.time_out_of_bed = 0
        else:
            if old_state in [self.STATE_IN_BED, self.STATE_LYING_DOWN]:
                self.current_state = self.STATE_GETTING_UP
                self.bed_exits += 1
            else:
                self.current_state = self.STATE_OUT_OF_BED
            self.time_out_of_bed += dt

        # Track state changes
        if old_state != self.current_state:
            self.previous_state = old_state
            self.state_start_time = current_time
            if self.current_state == self.STATE_IN_BED:
                self.bed_entries += 1

        return self.current_state

    def get_out_of_bed_duration(self):
        """Get how long person has been out of bed"""
        return self.time_out_of_bed

    def should_alert_out_of_bed(self):
        """Check if should alert for prolonged out of bed"""
        return (self.is_night_mode and
                self.current_state == self.STATE_OUT_OF_BED and
                self.time_out_of_bed > self.out_of_bed_threshold_sec)

    def get_statistics(self):
        """Get bed monitoring statistics"""
        return {
            "state": self.current_state,
            "time_in_bed_hours": round(self.time_in_bed_today / 3600, 2),
            "bed_entries": self.bed_entries,
            "bed_exits": self.bed_exits,
            "time_out_of_bed_sec": round(self.time_out_of_bed, 1)
        }


# ============================================================================
# VITAL SIGNS ANALYZER - Extended vital signs analysis
# ============================================================================
class VitalSignsAnalyzer:
    """Analyzes vital signs for anomalies and trends"""

    def __init__(self, history_minutes=30, frame_time_ms=55):
        self.frame_time = frame_time_ms
        self.history_len = int((history_minutes * 60 * 1000) / frame_time_ms)

        # History buffers
        self.heart_rate_history = deque(maxlen=self.history_len)
        self.breath_rate_history = deque(maxlen=self.history_len)
        self.timestamps = deque(maxlen=self.history_len)

        # Normal ranges
        self.hr_min = 50
        self.hr_max = 100
        self.br_min = 10
        self.br_max = 25

        # Current values
        self.current_hr = 0
        self.current_br = 0
        self.hr_status = "Normal"
        self.br_status = "Normal"

    def update(self, vitals_data):
        """Update with new vital signs data"""
        if vitals_data is None:
            return

        hr = vitals_data.get('heartRate', 0)
        br = vitals_data.get('breathRate', 0)

        if hr > 0:
            self.heart_rate_history.append(hr)
            self.current_hr = hr

        if br > 0:
            self.breath_rate_history.append(br)
            self.current_br = br

        self.timestamps.append(time.time())

        # Check status
        self._update_status()

    def _update_status(self):
        """Update vital signs status"""
        if self.current_hr < self.hr_min:
            self.hr_status = "Low"
        elif self.current_hr > self.hr_max:
            self.hr_status = "High"
        else:
            self.hr_status = "Normal"

        if self.current_br < self.br_min:
            self.br_status = "Low"
        elif self.current_br > self.br_max:
            self.br_status = "High"
        else:
            self.br_status = "Normal"

    def is_anomaly(self):
        """Check if current vitals are anomalous"""
        return self.hr_status != "Normal" or self.br_status != "Normal"

    def get_trends(self):
        """Get vital signs trends"""
        if len(self.heart_rate_history) < 10:
            return {"hr_trend": "Insufficient data", "br_trend": "Insufficient data"}

        recent_hr = list(self.heart_rate_history)[-50:]
        recent_br = list(self.breath_rate_history)[-50:]

        hr_slope = np.polyfit(range(len(recent_hr)), recent_hr, 1)[0] if len(recent_hr) > 1 else 0
        br_slope = np.polyfit(range(len(recent_br)), recent_br, 1)[0] if len(recent_br) > 1 else 0

        hr_trend = "Increasing" if hr_slope > 0.1 else "Decreasing" if hr_slope < -0.1 else "Stable"
        br_trend = "Increasing" if br_slope > 0.1 else "Decreasing" if br_slope < -0.1 else "Stable"

        return {"hr_trend": hr_trend, "br_trend": br_trend}

    def get_statistics(self):
        """Get vital signs statistics"""
        hr_list = list(self.heart_rate_history)
        br_list = list(self.breath_rate_history)

        return {
            "current_hr": round(self.current_hr, 1),
            "current_br": round(self.current_br, 1),
            "hr_status": self.hr_status,
            "br_status": self.br_status,
            "avg_hr": round(np.mean(hr_list), 1) if hr_list else 0,
            "avg_br": round(np.mean(br_list), 1) if br_list else 0,
            "hr_min_recorded": round(min(hr_list), 1) if hr_list else 0,
            "hr_max_recorded": round(max(hr_list), 1) if hr_list else 0
        }


# ============================================================================
# FALL RISK ANALYZER - Analyzes gait for fall risk
# ============================================================================
class FallRiskAnalyzer:
    """Analyzes movement patterns to assess fall risk"""

    def __init__(self, history_frames=500):
        self.history_len = history_frames

        # Movement patterns
        self.velocity_variance = deque(maxlen=history_frames)
        self.direction_changes = deque(maxlen=history_frames)
        self.height_variance = deque(maxlen=history_frames)

        # Risk score (0-100)
        self.fall_risk_score = 0
        self.risk_level = "Low"

        # Tracking
        self.last_velocity = None
        self.last_direction = None

    def update(self, tracks):
        """Update fall risk analysis"""
        if len(tracks) == 0:
            return

        track = tracks[0]
        vx, vy = track[4], track[5]
        z = track[3]

        velocity = np.sqrt(vx**2 + vy**2)
        direction = np.arctan2(vy, vx) if velocity > 0.1 else 0

        # Track variance
        if self.last_velocity is not None:
            vel_change = abs(velocity - self.last_velocity)
            self.velocity_variance.append(vel_change)

        if self.last_direction is not None and velocity > 0.1:
            dir_change = abs(direction - self.last_direction)
            if dir_change > np.pi:
                dir_change = 2 * np.pi - dir_change
            self.direction_changes.append(dir_change)

        self.height_variance.append(z)

        self.last_velocity = velocity
        self.last_direction = direction

        # Calculate risk score
        self._calculate_risk()

    def _calculate_risk(self):
        """Calculate fall risk score"""
        score = 0

        # High velocity variance = unstable walking
        if len(self.velocity_variance) > 10:
            vel_var = np.var(list(self.velocity_variance))
            score += min(30, vel_var * 100)

        # Frequent direction changes = shuffling/instability
        if len(self.direction_changes) > 10:
            dir_var = np.var(list(self.direction_changes))
            score += min(30, dir_var * 20)

        # Height variance = swaying
        if len(self.height_variance) > 10:
            h_var = np.var(list(self.height_variance))
            score += min(40, h_var * 200)

        self.fall_risk_score = min(100, score)

        # Determine risk level
        if self.fall_risk_score < 30:
            self.risk_level = "Low"
        elif self.fall_risk_score < 60:
            self.risk_level = "Medium"
        else:
            self.risk_level = "High"

    def get_statistics(self):
        """Get fall risk statistics"""
        return {
            "fall_risk_score": round(self.fall_risk_score, 1),
            "risk_level": self.risk_level
        }


# ============================================================================
# PRESENCE TRACKER - Tracks room presence and visitor detection
# ============================================================================
class PresenceTracker:
    """Tracks presence, visitors, and loneliness metrics"""

    def __init__(self, room_zone, frame_time_ms=55):
        self.room_zone = room_zone
        self.frame_time = frame_time_ms

        # Presence tracking
        self.people_count = 0
        self.time_alone_today = 0  # seconds
        self.time_with_visitors = 0
        self.visitor_count_today = 0

        # State
        self.is_alone = True
        self.last_update = time.time()
        self.person_in_room = False

        # Loneliness calculation
        self.loneliness_score = 0  # 0-100, higher = more lonely

    def update(self, tracks):
        """Update presence tracking"""
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time

        # Count people in room
        people_in_room = 0
        for track in tracks:
            x, y = track[1], track[2]
            if self.room_zone.contains(x, y):
                people_in_room += 1

        old_count = self.people_count
        self.people_count = people_in_room
        self.person_in_room = people_in_room > 0

        # Track time alone vs with visitors
        if people_in_room <= 1:
            self.is_alone = True
            self.time_alone_today += dt
        else:
            self.is_alone = False
            self.time_with_visitors += dt
            if old_count == 1 and people_in_room > 1:
                self.visitor_count_today += 1

        # Calculate loneliness score
        self._calculate_loneliness()

    def _calculate_loneliness(self):
        """Calculate loneliness score (0-100)"""
        total_time = self.time_alone_today + self.time_with_visitors
        if total_time < 60:  # Need at least 1 minute of data
            self.loneliness_score = 0
            return

        # Base score on time alone
        alone_ratio = self.time_alone_today / total_time
        self.loneliness_score = alone_ratio * 80

        # Reduce score for visitor count
        self.loneliness_score -= min(30, self.visitor_count_today * 10)
        self.loneliness_score = max(0, min(100, self.loneliness_score))

    def is_out_of_room(self, tracks):
        """Check if primary person left the room"""
        if len(tracks) == 0:
            return True
        track = tracks[0]
        return not self.room_zone.contains(track[1], track[2])

    def get_statistics(self):
        """Get presence statistics"""
        return {
            "people_in_room": self.people_count,
            "is_alone": self.is_alone,
            "time_alone_hours": round(self.time_alone_today / 3600, 2),
            "visitor_count": self.visitor_count_today,
            "loneliness_score": round(self.loneliness_score, 1)
        }


# ============================================================================
# MAIN CARE MONITORING CLASS
# ============================================================================
class CareMonitoring(Plot3D, Plot1D):
    """
    Main Care Monitoring Demo Class

    Features:
    - Fall Detection (from TI SDK)
    - Vital Signs Monitoring
    - Bed Occupancy Tracking
    - Activity Level Monitoring
    - Room Presence Detection
    - Fall Risk Analysis
    - Alert System
    - Data Logging
    """

    def __init__(self):
        Plot3D.__init__(self)
        Plot1D.__init__(self)

        # Core TI components
        self.fallDetection = FallDetection()
        self.tabs = None
        self.cumulativeCloud = None
        self.colorGradient = pg.GradientWidget(orientation='right')
        self.colorGradient.restoreState({
            'ticks': [(1, (255, 0, 0, 255)), (0, (131, 238, 255, 255))],
            'mode': 'hsv'
        })
        self.colorGradient.setVisible(False)
        self.maxTracks = 5
        self.frameTime = 55
        self.trackColorMap = get_trackColors(self.maxTracks)

        # Care monitoring components
        self.alert_system = AlertSystem()
        self.alert_system.setup_logging()

        # Default zones (configurable via GUI)
        self.bed_zone = Zone("Bed", (-1, 1), (0, 2), (0, 1))
        self.room_zone = Zone("Room", (-3, 3), (-1, 5), (-1, 3))
        self.bathroom_zone = Zone("Bathroom", (-3, -1), (3, 5), (-1, 3))

        # Analyzers
        self.activity_tracker = ActivityTracker(frame_time_ms=self.frameTime)
        self.bed_monitor = BedMonitor(self.bed_zone, frame_time_ms=self.frameTime)
        self.vitals_analyzer = VitalSignsAnalyzer(frame_time_ms=self.frameTime)
        self.fall_risk_analyzer = FallRiskAnalyzer()
        self.presence_tracker = PresenceTracker(self.room_zone, frame_time_ms=self.frameTime)

        # Vitals data storage
        self.vitalsDict = None
        self.numTracks = None
        self.vitalsPatientData = []

        # Alert flags (to prevent repeated alerts)
        self.fall_alert_active = False
        self.out_of_bed_alert_active = False
        self.vital_alert_active = False
        self.out_of_room_alert_active = False

        # Statistics update timer
        self.stats_update_counter = 0

    def setupGUI(self, gridLayout, demoTabs, device):
        """Setup the GUI for Care Monitoring"""

        # Statistics pane
        statBox = self.initStatsPane()
        gridLayout.addWidget(statBox, 2, 0, 1, 1)

        # Plot controls
        plotControlBox = self.initPlotControlPane()
        gridLayout.addWidget(plotControlBox, 3, 0, 1, 1)

        # Fall detection controls
        fallDetBox = self.initFallDetectPane()
        gridLayout.addWidget(fallDetBox, 4, 0, 1, 1)

        # Zone configuration
        zoneConfigBox = self.initZoneConfigPane()
        gridLayout.addWidget(zoneConfigBox, 5, 0, 1, 1)

        # Add tabs
        demoTabs.addTab(self.plot_3d, '3D View')
        demoTabs.addTab(self.rangePlot, 'Range Plot')

        # Care Dashboard tab
        self.careDashboard = self.initCareDashboard()
        demoTabs.addTab(self.careDashboard, 'Care Dashboard')

        # Vitals tab
        self.vitalsPane = self.initVitalsPane()
        demoTabs.addTab(self.vitalsPane, 'Vital Signs')

        # Alerts tab
        self.alertsPane = self.initAlertsPane()
        demoTabs.addTab(self.alertsPane, 'Alerts Log')

        self.device = device
        self.tabs = demoTabs

        # Connect alert signal
        self.alert_system.alertTriggered.connect(self.onAlertTriggered)

    def initStatsPane(self):
        """Initialize statistics display pane"""
        statBox = QGroupBox('Statistics')
        self.frameNumDisplay = QLabel('Frame: 0')
        self.plotTimeDisplay = QLabel('Plot Time: 0 ms')
        self.numPointsDisplay = QLabel('Points: 0')
        self.numTargetsDisplay = QLabel('Targets: 0')
        self.avgPower = QLabel('Average Power: 0 mW')

        self.statsLayout = QVBoxLayout()
        self.statsLayout.addWidget(self.frameNumDisplay)
        self.statsLayout.addWidget(self.plotTimeDisplay)
        self.statsLayout.addWidget(self.numPointsDisplay)
        self.statsLayout.addWidget(self.numTargetsDisplay)
        self.statsLayout.addWidget(self.avgPower)
        statBox.setLayout(self.statsLayout)
        return statBox

    def initPlotControlPane(self):
        """Initialize plot control pane"""
        plotControlBox = QGroupBox('Plot Controls')

        from Demo_Classes.people_tracking import COLOR_MODE_SNR, COLOR_MODE_HEIGHT, COLOR_MODE_DOPPLER, COLOR_MODE_TRACK, MAX_PERSISTENT_FRAMES

        self.pointColorMode = QComboBox()
        self.pointColorMode.addItems([COLOR_MODE_SNR, COLOR_MODE_HEIGHT, COLOR_MODE_DOPPLER, COLOR_MODE_TRACK])

        self.displayFallDet = QCheckBox('Enable Fall Detection')
        self.displayFallDet.setChecked(True)
        self.displayFallDet.stateChanged.connect(self.fallDetDisplayChanged)

        self.persistentFramesInput = QComboBox()
        self.persistentFramesInput.addItems([str(i) for i in range(1, MAX_PERSISTENT_FRAMES + 1)])
        self.persistentFramesInput.setCurrentIndex(self.numPersistentFrames - 1)
        self.persistentFramesInput.currentIndexChanged.connect(self.persistentFramesChanged)

        plotControlLayout = QFormLayout()
        plotControlLayout.addRow("Color Points By:", self.pointColorMode)
        plotControlLayout.addRow("Fall Detection:", self.displayFallDet)
        plotControlLayout.addRow("Persistent Frames:", self.persistentFramesInput)
        plotControlBox.setLayout(plotControlLayout)

        return plotControlBox

    def initFallDetectPane(self):
        """Initialize fall detection sensitivity pane"""
        self.fallDetectionOptionsBox = QGroupBox('Fall Detection Sensitivity')
        self.fallDetLayout = QGridLayout()

        self.fallDetSlider = FallDetectionSliderClass(Qt.Horizontal)
        self.fallDetSlider.setTracking(True)
        self.fallDetSlider.setTickPosition(QSlider.TicksBothSides)
        self.fallDetSlider.setTickInterval(10)
        self.fallDetSlider.setRange(0, 100)
        self.fallDetSlider.setSliderPosition(50)
        self.fallDetSlider.valueChanged.connect(self.updateFallDetectionSensitivity)

        self.fallDetLayout.addWidget(QLabel("Less Sensitive"), 0, 0, 1, 1)
        self.fallDetLayout.addWidget(QLabel("More Sensitive"), 0, 10, 1, 1)
        self.fallDetLayout.addWidget(self.fallDetSlider, 1, 0, 1, 11)
        self.fallDetectionOptionsBox.setLayout(self.fallDetLayout)

        return self.fallDetectionOptionsBox

    def initZoneConfigPane(self):
        """Initialize zone configuration pane"""
        zoneBox = QGroupBox('Zone Configuration')
        layout = QFormLayout()

        # Bed zone
        self.bedXMin = QDoubleSpinBox()
        self.bedXMin.setRange(-10, 10)
        self.bedXMin.setValue(self.bed_zone.x_min)
        self.bedXMax = QDoubleSpinBox()
        self.bedXMax.setRange(-10, 10)
        self.bedXMax.setValue(self.bed_zone.x_max)
        self.bedYMin = QDoubleSpinBox()
        self.bedYMin.setRange(-10, 10)
        self.bedYMin.setValue(self.bed_zone.y_min)
        self.bedYMax = QDoubleSpinBox()
        self.bedYMax.setRange(-10, 10)
        self.bedYMax.setValue(self.bed_zone.y_max)

        bedLayout = QHBoxLayout()
        bedLayout.addWidget(QLabel("X:"))
        bedLayout.addWidget(self.bedXMin)
        bedLayout.addWidget(QLabel("to"))
        bedLayout.addWidget(self.bedXMax)
        bedLayout.addWidget(QLabel("Y:"))
        bedLayout.addWidget(self.bedYMin)
        bedLayout.addWidget(QLabel("to"))
        bedLayout.addWidget(self.bedYMax)
        layout.addRow("Bed Zone:", bedLayout)

        # Apply button
        applyBtn = QPushButton("Apply Zones")
        applyBtn.clicked.connect(self.applyZoneConfig)
        layout.addRow(applyBtn)

        # Night mode
        self.nightModeCheck = QCheckBox("Night Mode (enhanced out-of-bed alerts)")
        self.nightModeCheck.stateChanged.connect(self.toggleNightMode)
        layout.addRow(self.nightModeCheck)

        zoneBox.setLayout(layout)
        return zoneBox

    def initCareDashboard(self):
        """Initialize care monitoring dashboard"""
        dashboard = QWidget()
        layout = QGridLayout()

        # Style for labels
        headerFont = QFont('Arial', 14, QFont.Bold)
        valueFont = QFont('Arial', 12)

        # Fall Status
        fallGroup = QGroupBox("Fall Detection")
        fallLayout = QVBoxLayout()
        self.fallStatusLabel = QLabel("No Fall Detected")
        self.fallStatusLabel.setFont(headerFont)
        self.fallStatusLabel.setStyleSheet("color: green;")
        fallLayout.addWidget(self.fallStatusLabel)
        fallGroup.setLayout(fallLayout)
        layout.addWidget(fallGroup, 0, 0)

        # Bed Status
        bedGroup = QGroupBox("Bed Monitor")
        bedLayout = QVBoxLayout()
        self.bedStatusLabel = QLabel("Status: Unknown")
        self.bedStatusLabel.setFont(valueFont)
        self.timeInBedLabel = QLabel("Time in Bed: 0 h")
        self.bedExitsLabel = QLabel("Bed Exits: 0")
        bedLayout.addWidget(self.bedStatusLabel)
        bedLayout.addWidget(self.timeInBedLabel)
        bedLayout.addWidget(self.bedExitsLabel)
        bedGroup.setLayout(bedLayout)
        layout.addWidget(bedGroup, 0, 1)

        # Activity
        activityGroup = QGroupBox("Activity Level")
        activityLayout = QVBoxLayout()
        self.activityLevelLabel = QLabel("Level: Unknown")
        self.activityLevelLabel.setFont(valueFont)
        self.activityScoreLabel = QLabel("Score: 0")
        self.distanceLabel = QLabel("Distance: 0 m")
        activityLayout.addWidget(self.activityLevelLabel)
        activityLayout.addWidget(self.activityScoreLabel)
        activityLayout.addWidget(self.distanceLabel)
        activityGroup.setLayout(activityLayout)
        layout.addWidget(activityGroup, 0, 2)

        # Fall Risk
        riskGroup = QGroupBox("Fall Risk")
        riskLayout = QVBoxLayout()
        self.fallRiskLabel = QLabel("Risk: Low")
        self.fallRiskLabel.setFont(valueFont)
        self.fallRiskScoreLabel = QLabel("Score: 0")
        riskLayout.addWidget(self.fallRiskLabel)
        riskLayout.addWidget(self.fallRiskScoreLabel)
        riskGroup.setLayout(riskLayout)
        layout.addWidget(riskGroup, 1, 0)

        # Presence
        presenceGroup = QGroupBox("Presence & Social")
        presenceLayout = QVBoxLayout()
        self.peopleCountLabel = QLabel("People in Room: 0")
        self.peopleCountLabel.setFont(valueFont)
        self.lonelinessLabel = QLabel("Loneliness Score: 0")
        self.visitorsLabel = QLabel("Visitors Today: 0")
        presenceLayout.addWidget(self.peopleCountLabel)
        presenceLayout.addWidget(self.lonelinessLabel)
        presenceLayout.addWidget(self.visitorsLabel)
        presenceGroup.setLayout(presenceLayout)
        layout.addWidget(presenceGroup, 1, 1)

        # Vitals Summary
        vitalsGroup = QGroupBox("Vital Signs")
        vitalsLayout = QVBoxLayout()
        self.heartRateLabel = QLabel("Heart Rate: -- bpm")
        self.heartRateLabel.setFont(valueFont)
        self.breathRateLabel = QLabel("Breath Rate: -- rpm")
        self.vitalsStatusLabel = QLabel("Status: Normal")
        vitalsLayout.addWidget(self.heartRateLabel)
        vitalsLayout.addWidget(self.breathRateLabel)
        vitalsLayout.addWidget(self.vitalsStatusLabel)
        vitalsGroup.setLayout(vitalsLayout)
        layout.addWidget(vitalsGroup, 1, 2)

        dashboard.setLayout(layout)
        return dashboard

    def initVitalsPane(self):
        """Initialize vital signs display pane"""
        vitalsPane = QWidget()
        layout = QVBoxLayout()

        # Heart rate plot
        self.heartPlot = pg.PlotWidget(title="Heart Rate")
        self.heartPlot.setBackground('w')
        self.heartPlot.showGrid(x=True, y=True)
        self.heartPlot.setYRange(40, 120)
        self.heartPlot.setLabel('left', 'BPM')
        self.heartPlot.setLabel('bottom', 'Time')
        self.heartCurve = self.heartPlot.plot(pen=pg.mkPen('r', width=2))
        layout.addWidget(self.heartPlot)

        # Breath rate plot
        self.breathPlot = pg.PlotWidget(title="Breathing Rate")
        self.breathPlot.setBackground('w')
        self.breathPlot.showGrid(x=True, y=True)
        self.breathPlot.setYRange(5, 35)
        self.breathPlot.setLabel('left', 'RPM')
        self.breathPlot.setLabel('bottom', 'Time')
        self.breathCurve = self.breathPlot.plot(pen=pg.mkPen('b', width=2))
        layout.addWidget(self.breathPlot)

        vitalsPane.setLayout(layout)
        return vitalsPane

    def initAlertsPane(self):
        """Initialize alerts log pane"""
        alertsPane = QWidget()
        layout = QVBoxLayout()

        # Alert log
        self.alertLog = QTextEdit()
        self.alertLog.setReadOnly(True)
        self.alertLog.setStyleSheet("font-family: monospace;")
        layout.addWidget(QLabel("Alert History:"))
        layout.addWidget(self.alertLog)

        # Clear button
        clearBtn = QPushButton("Clear Alerts")
        clearBtn.clicked.connect(lambda: self.alertLog.clear())
        layout.addWidget(clearBtn)

        alertsPane.setLayout(layout)
        return alertsPane

    def updateGraph(self, outputDict):
        """Main update function - called every frame"""
        self.plotStart = int(round(time.time() * 1000))
        self.updatePointCloud(outputDict)

        self.cumulativeCloud = None

        # Build cumulative point cloud
        if ('frameNum' in outputDict and outputDict['frameNum'] > 1 and
            len(self.previousClouds[:-1]) > 0 and DEVICE_DEMO_DICT[self.device]["isxWRx843"]):
            for frame in range(len(self.previousClouds[:-1])):
                if len(self.previousClouds[frame]) > 0:
                    if self.cumulativeCloud is None:
                        self.cumulativeCloud = self.previousClouds[frame]
                    else:
                        self.cumulativeCloud = np.concatenate(
                            (self.cumulativeCloud, self.previousClouds[frame]), axis=0)
        elif len(self.previousClouds) > 0:
            for frame in range(len(self.previousClouds[:])):
                if len(self.previousClouds[frame]) > 0:
                    if self.cumulativeCloud is None:
                        self.cumulativeCloud = self.previousClouds[frame]
                    else:
                        self.cumulativeCloud = np.concatenate(
                            (self.cumulativeCloud, self.previousClouds[frame]), axis=0)

        # Update statistics displays
        if 'numDetectedPoints' in outputDict:
            self.numPointsDisplay.setText('Points: ' + str(outputDict['numDetectedPoints']))
        if 'numDetectedTracks' in outputDict:
            self.numTargetsDisplay.setText('Targets: ' + str(outputDict['numDetectedTracks']))

        # Hide all track labels
        for cstr in self.coordStr:
            cstr.setVisible(False)

        # Process tracks and care monitoring
        tracks = None
        if 'trackData' in outputDict:
            tracks = outputDict['trackData']
            num_tracks = outputDict.get('numDetectedTracks', 0)

            # Apply rotations
            for i in range(num_tracks):
                rotX, rotY, rotZ = eulerRot(tracks[i, 1], tracks[i, 2], tracks[i, 3],
                                           self.elev_tilt, self.az_tilt)
                tracks[i, 1] = rotX
                tracks[i, 2] = rotY
                tracks[i, 3] = rotZ + self.sensorHeight

            # Update care monitoring systems
            self.updateCareMonitoring(tracks, outputDict)

            # Process height data and fall detection
            if 'heightData' in outputDict:
                for height in outputDict['heightData']:
                    for track in outputDict['trackData']:
                        if int(track[0]) == int(height[0]):
                            tid = int(height[0])
                            height_str = f'tid: {height[0]}, height: {round(height[1], 2)} m'

                            # Fall detection
                            if self.displayFallDet.checkState() == 2:
                                fallResults = self.fallDetection.step(
                                    outputDict['heightData'], outputDict['trackData'])
                                if fallResults[tid] > 0:
                                    height_str += " FALL DETECTED"
                                    self.onFallDetected(tid)
                                else:
                                    self.fall_alert_active = False

                            self.coordStr[tid].setText(height_str)
                            self.coordStr[tid].setX(track[1])
                            self.coordStr[tid].setY(track[2])
                            self.coordStr[tid].setZ(track[3])
                            self.coordStr[tid].setVisible(True)
                            break

        # Update vital signs
        if 'vitals' in outputDict:
            self.updateVitals(outputDict['vitals'])

        # 3D Plot update
        if self.tabs.currentWidget() == self.plot_3d:
            if self.plotComplete:
                self.plotStart = int(round(time.time() * 1000))
                self.plot_3d_thread = updateQTTargetThread3D(
                    self.cumulativeCloud, tracks, self.scatter, self.plot_3d, 0,
                    self.ellipsoids, "", colorGradient=self.colorGradient,
                    pointColorMode=self.pointColorMode.currentText(),
                    trackColorMap=self.trackColorMap)
                self.plotComplete = 0
                self.plot_3d_thread.done.connect(lambda: self.graphDone(outputDict))
                self.plot_3d_thread.start(priority=QThread.HighPriority)
        elif self.tabs.currentWidget() == self.rangePlot:
            self.update1DGraph(outputDict)
            self.graphDone(outputDict)
        else:
            self.graphDone(outputDict)

        if 'frameNum' in outputDict:
            self.frameNumDisplay.setText('Frame: ' + str(outputDict['frameNum']))

    def updateCareMonitoring(self, tracks, outputDict):
        """Update all care monitoring systems"""
        if tracks is None or len(tracks) == 0:
            return

        # Activity tracking
        self.activity_tracker.update(tracks)

        # Bed monitoring
        bed_state = self.bed_monitor.update(tracks, outputDict.get('heightData'))

        # Fall risk analysis
        self.fall_risk_analyzer.update(tracks)

        # Presence tracking
        self.presence_tracker.update(tracks)

        # Check for alerts
        self.checkAlerts(tracks)

        # Update dashboard (every 10 frames to reduce load)
        self.stats_update_counter += 1
        if self.stats_update_counter >= 10:
            self.stats_update_counter = 0
            self.updateDashboard()

    def updateVitals(self, vitals_data):
        """Update vital signs displays"""
        self.vitals_analyzer.update(vitals_data)

        # Update plots
        hr_data = list(self.vitals_analyzer.heart_rate_history)
        br_data = list(self.vitals_analyzer.breath_rate_history)

        if len(hr_data) > 1:
            self.heartCurve.setData(hr_data)
        if len(br_data) > 1:
            self.breathCurve.setData(br_data)

        # Check for vital sign anomalies
        if self.vitals_analyzer.is_anomaly() and not self.vital_alert_active:
            self.vital_alert_active = True
            stats = self.vitals_analyzer.get_statistics()
            self.alert_system.trigger_alert(
                "VITAL_ANOMALY",
                f"Heart Rate: {stats['current_hr']} ({stats['hr_status']}), "
                f"Breath Rate: {stats['current_br']} ({stats['br_status']})",
                "WARNING"
            )
        elif not self.vitals_analyzer.is_anomaly():
            self.vital_alert_active = False

    def updateDashboard(self):
        """Update care dashboard displays"""
        # Activity
        activity_stats = self.activity_tracker.get_statistics()
        self.activityLevelLabel.setText(f"Level: {activity_stats['activity_level']}")
        self.activityScoreLabel.setText(f"Score: {round(activity_stats['activity_score'], 1)}")
        self.distanceLabel.setText(f"Distance: {activity_stats['total_distance_m']} m")

        # Bed
        bed_stats = self.bed_monitor.get_statistics()
        self.bedStatusLabel.setText(f"Status: {bed_stats['state']}")
        self.timeInBedLabel.setText(f"Time in Bed: {bed_stats['time_in_bed_hours']} h")
        self.bedExitsLabel.setText(f"Bed Exits: {bed_stats['bed_exits']}")

        # Fall Risk
        risk_stats = self.fall_risk_analyzer.get_statistics()
        self.fallRiskLabel.setText(f"Risk: {risk_stats['risk_level']}")
        self.fallRiskScoreLabel.setText(f"Score: {risk_stats['fall_risk_score']}")

        # Color code fall risk
        if risk_stats['risk_level'] == "High":
            self.fallRiskLabel.setStyleSheet("color: red;")
        elif risk_stats['risk_level'] == "Medium":
            self.fallRiskLabel.setStyleSheet("color: orange;")
        else:
            self.fallRiskLabel.setStyleSheet("color: green;")

        # Presence
        presence_stats = self.presence_tracker.get_statistics()
        self.peopleCountLabel.setText(f"People in Room: {presence_stats['people_in_room']}")
        self.lonelinessLabel.setText(f"Loneliness Score: {presence_stats['loneliness_score']}")
        self.visitorsLabel.setText(f"Visitors Today: {presence_stats['visitor_count']}")

        # Vitals
        vital_stats = self.vitals_analyzer.get_statistics()
        self.heartRateLabel.setText(f"Heart Rate: {vital_stats['current_hr']} bpm")
        self.breathRateLabel.setText(f"Breath Rate: {vital_stats['current_br']} rpm")
        self.vitalsStatusLabel.setText(f"Status: HR {vital_stats['hr_status']}, BR {vital_stats['br_status']}")

    def checkAlerts(self, tracks):
        """Check for alert conditions"""
        # Out of bed alert (night mode)
        if self.bed_monitor.should_alert_out_of_bed() and not self.out_of_bed_alert_active:
            self.out_of_bed_alert_active = True
            duration = self.bed_monitor.get_out_of_bed_duration()
            self.alert_system.trigger_alert(
                "OUT_OF_BED",
                f"Person has been out of bed for {round(duration, 0)} seconds",
                "WARNING"
            )
        elif not self.bed_monitor.should_alert_out_of_bed():
            self.out_of_bed_alert_active = False

        # Out of room alert
        if self.presence_tracker.is_out_of_room(tracks) and not self.out_of_room_alert_active:
            self.out_of_room_alert_active = True
            self.alert_system.trigger_alert(
                "OUT_OF_ROOM",
                "Person has left the monitored room",
                "INFO"
            )
        elif not self.presence_tracker.is_out_of_room(tracks):
            self.out_of_room_alert_active = False

    def onFallDetected(self, track_id):
        """Handle fall detection event"""
        if not self.fall_alert_active:
            self.fall_alert_active = True
            self.fallStatusLabel.setText("FALL DETECTED!")
            self.fallStatusLabel.setStyleSheet("color: red; font-weight: bold;")
            self.alert_system.trigger_alert(
                "FALL_DETECTED",
                f"Fall detected for person {track_id}",
                "CRITICAL"
            )

            # Reset after 5 seconds
            QTimer.singleShot(5000, self.resetFallStatus)

    def resetFallStatus(self):
        """Reset fall status display"""
        self.fallStatusLabel.setText("No Fall Detected")
        self.fallStatusLabel.setStyleSheet("color: green;")

    def onAlertTriggered(self, alert_type, message, timestamp):
        """Handle alert signal"""
        color = "black"
        if "CRITICAL" in alert_type or "FALL" in alert_type:
            color = "red"
        elif "WARNING" in alert_type:
            color = "orange"

        self.alertLog.append(f'<span style="color:{color}">[{timestamp}] {alert_type}: {message}</span>')

    def graphDone(self, outputDict):
        """Called when graph update is complete"""
        if 'frameNum' in outputDict:
            self.frameNumDisplay.setText('Frame: ' + str(outputDict['frameNum']))

        if 'powerData' in outputDict:
            self.updatePowerNumbers(outputDict['powerData'])

        plotTime = int(round(time.time() * 1000)) - self.plotStart
        self.plotTimeDisplay.setText('Plot Time: ' + str(plotTime) + 'ms')
        self.plotComplete = 1

    def updatePowerNumbers(self, powerData):
        """Update power display"""
        if powerData['power1v2'] == 65535:
            self.avgPower.setText('Average Power: N/A')
        else:
            powerStr = str((powerData['power1v2'] + powerData['power1v2RF'] +
                          powerData['power1v8'] + powerData['power3v3']) * 0.1)
            self.avgPower.setText('Average Power: ' + powerStr[:5] + ' mW')

    # =========== Configuration Methods ===========

    def persistentFramesChanged(self, index):
        """Handle persistent frames change"""
        self.numPersistentFrames = index + 1

    def fallDetDisplayChanged(self, state):
        """Handle fall detection toggle"""
        self.fallDetectionOptionsBox.setVisible(state == 2)

    def updateFallDetectionSensitivity(self):
        """Update fall detection sensitivity"""
        value = self.fallDetSlider.value() / self.fallDetSlider.maximum()
        sensitivity = (value * 0.4) + 0.4  # Range 0.4 to 0.8
        self.fallDetection.setFallSensitivity(sensitivity)

    def applyZoneConfig(self):
        """Apply zone configuration changes"""
        self.bed_zone = Zone(
            "Bed",
            (self.bedXMin.value(), self.bedXMax.value()),
            (self.bedYMin.value(), self.bedYMax.value())
        )
        self.bed_monitor.bed_zone = self.bed_zone
        log.info(f"Bed zone updated: {self.bed_zone.to_dict()}")

    def toggleNightMode(self, state):
        """Toggle night mode for bed monitoring"""
        self.bed_monitor.is_night_mode = (state == 2)
        log.info(f"Night mode: {self.bed_monitor.is_night_mode}")

    def parseTrackingCfg(self, args):
        """Parse tracking configuration from cfg file"""
        self.maxTracks = int(args[4])
        if len(args) == 8:
            self.frameTime = int(args[7])
        else:
            self.frameTime = None

        self.updateNumTracksBuffer()
        self.trackColorMap = get_trackColors(self.maxTracks)

        for m in range(self.maxTracks):
            # Add track mesh
            mesh = gl.GLLinePlotItem()
            mesh.setVisible(False)
            self.plot_3d.addItem(mesh)
            self.ellipsoids.append(mesh)

            # Add track label
            text = GLTextItem()
            text.setGLViewWidget(self.plot_3d)
            text.setVisible(False)
            self.plot_3d.addItem(text)
            self.coordStr.append(text)

            # Add classifier label
            classifierText = GLTextItem()
            classifierText.setGLViewWidget(self.plot_3d)
            classifierText.setVisible(False)
            self.plot_3d.addItem(classifierText)
            self.classifierStr.append(classifierText)

    def updateNumTracksBuffer(self):
        """Update fall detection buffer for track count"""
        if self.frameTime is not None:
            self.fallDetection = FallDetection(self.maxTracks, self.frameTime)
            self.activity_tracker = ActivityTracker(frame_time_ms=self.frameTime)
            self.bed_monitor = BedMonitor(self.bed_zone, frame_time_ms=self.frameTime)
        else:
            self.fallDetection = FallDetection(self.maxTracks)
