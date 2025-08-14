"""
Historical Data Manager for Temporal Anomaly Detection

This module provides a thread-safe in-memory storage for recent readings
to support sequence-based predictions in LSTM and temporal Isolation Forest models.
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import numpy as np

class HistoricalDataManager:
    def __init__(self, max_sequence_length=20, max_age_hours=24):
        """
        Initialize the historical data manager.
        
        Args:
            max_sequence_length (int): Maximum number of readings to store per profile
            max_age_hours (int): Maximum age of readings to keep (in hours)
        """
        self.data = defaultdict(lambda: deque(maxlen=max_sequence_length))
        self.max_age = timedelta(hours=max_age_hours)
        self.lock = threading.Lock()
        self.max_sequence_length = max_sequence_length
    
    def add_reading(self, profile_id, timestamp, features):
        """
        Add a new reading to the historical data.
        
        Args:
            profile_id (str): The profile identifier
            timestamp (datetime): The timestamp of the reading
            features (list): The feature values (voltage, current, power, frequency, power_factor)
        """
        with self.lock:
            # Ensure timestamp is datetime object
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Convert features to numpy array for consistency
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            self.data[profile_id].append({
                'timestamp': timestamp,
                'features': features
            })
            
            # Clean up old data for this profile
            self._cleanup_old_data(profile_id)
    
    def get_sequence(self, profile_id, length=10):
        """
        Get the most recent sequence of readings for a profile.
        
        Args:
            profile_id (str): The profile identifier
            length (int): Number of recent readings to retrieve
            
        Returns:
            list: List of dictionaries with 'timestamp' and 'features' keys,
                  or None if insufficient data
        """
        with self.lock:
            profile_data = self.data.get(profile_id, deque())
            
            if len(profile_data) < length:
                return None
            
            return list(profile_data)[-length:]
    
    def get_all_readings(self, profile_id):
        """
        Get all stored readings for a profile.
        
        Args:
            profile_id (str): The profile identifier
            
        Returns:
            list: List of all readings for the profile
        """
        with self.lock:
            profile_data = self.data.get(profile_id, deque())
            return list(profile_data)
    
    def get_profile_count(self):
        """Get the number of profiles with stored data."""
        with self.lock:
            return len(self.data)
    
    def get_reading_count(self, profile_id):
        """Get the number of readings stored for a specific profile."""
        with self.lock:
            return len(self.data.get(profile_id, deque()))
    
    def clear_profile(self, profile_id):
        """Clear all data for a specific profile."""
        with self.lock:
            if profile_id in self.data:
                self.data[profile_id].clear()
    
    def clear_all(self):
        """Clear all stored data."""
        with self.lock:
            self.data.clear()
    
    def _cleanup_old_data(self, profile_id):
        """
        Remove readings older than max_age for a specific profile.
        
        Args:
            profile_id (str): The profile identifier
        """
        cutoff_time = datetime.now() - self.max_age
        profile_data = self.data[profile_id]
        
        # Remove old readings from the left (oldest first)
        while profile_data and profile_data[0]['timestamp'] < cutoff_time:
            profile_data.popleft()
    
    def cleanup_all_old_data(self):
        """Remove old readings from all profiles."""
        with self.lock:
            for profile_id in list(self.data.keys()):
                self._cleanup_old_data(profile_id)
                
                # Remove empty profiles
                if not self.data[profile_id]:
                    del self.data[profile_id]
    
    def get_stats(self):
        """
        Get statistics about stored data.
        
        Returns:
            dict: Statistics including profile count, total readings, etc.
        """
        with self.lock:
            total_readings = sum(len(readings) for readings in self.data.values())
            profile_stats = {}
            
            for profile_id, readings in self.data.items():
                if readings:
                    oldest = min(r['timestamp'] for r in readings)
                    newest = max(r['timestamp'] for r in readings)
                    profile_stats[profile_id] = {
                        'count': len(readings),
                        'oldest': oldest.isoformat(),
                        'newest': newest.isoformat()
                    }
            
            return {
                'total_profiles': len(self.data),
                'total_readings': total_readings,
                'max_sequence_length': self.max_sequence_length,
                'max_age_hours': self.max_age.total_seconds() / 3600,
                'profiles': profile_stats
            }

# Global instance
historical_manager = HistoricalDataManager(max_sequence_length=20, max_age_hours=24)
