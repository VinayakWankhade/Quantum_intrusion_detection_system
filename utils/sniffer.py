import time
from scapy.all import sniff, IP, TCP, UDP, ICMP
import pandas as pd
import joblib
import os
import threading
from utils.helpers import get_logger

logger = get_logger(__name__)

class LiveFeatureExtractor:
    def __init__(self):
        pre_dir = "models/saved/preprocessing"
        self.scaler = joblib.load(os.path.join(pre_dir, "scaler_unified.pkl"))
        self.pca = joblib.load(os.path.join(pre_dir, "pca_unified.pkl"))
        self.feature_names = joblib.load(os.path.join(pre_dir, "feature_names_unified.pkl"))
        
        # Unified base features for mapping
        self.base_columns = ['duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'count', 'srv_count']

    def extract_from_packet(self, pkt):
        # Default empty feature dict
        features = {col: 0 for col in self.base_columns}
        
        if IP in pkt:
            features['src_bytes'] = len(pkt[IP].payload)
            # Simple mapping for internal logic
            if TCP in pkt:
                features['protocol_type'] = 'tcp'
                features['service'] = str(pkt[TCP].dport)
                features['flag'] = 'SF' # Default to Normal
            elif UDP in pkt:
                features['protocol_type'] = 'udp'
                features['service'] = str(pkt[UDP].dport)
                features['flag'] = 'SF'
            elif ICMP in pkt:
                features['protocol_type'] = 'icmp'
                features['service'] = 'eco_i'
                features['flag'] = 'SF'
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # 1. Manual One-Hot Encode (for speed)
        df['protocol_type_tcp'] = 1 if features['protocol_type'] == 'tcp' else 0
        df['protocol_type_udp'] = 1 if features['protocol_type'] == 'udp' else 0
        df['protocol_type_icmp'] = 1 if features['protocol_type'] == 'icmp' else 0
        df.drop(columns=['protocol_type'], inplace=True)
        
        # Ensure all expected columns exist
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder to match training
        df = df[self.feature_names]
        
        # 2. Scale
        X_scaled = self.scaler.transform(df)
        
        # 3. PCA
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca

class PacketSniffer:
    def __init__(self, callback):
        self.callback = callback
        self.extractor = LiveFeatureExtractor()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_sniffer, daemon=True)
        self.thread.start()
        logger.info("Sniffer thread started.")

    def stop(self):
        self.running = False
        logger.info("Sniffer thread stopping...")

    def _run_sniffer(self):
        try:
            sniff(prn=self._process_packet, stop_filter=lambda x: not self.running, store=0)
        except Exception as e:
            logger.error(f"Sniffer error: {e}")

    def _process_packet(self, pkt):
        if not (IP in pkt): return
        
        try:
            X_pca = self.extractor.extract_from_packet(pkt)
            
            # Metadata for UI
            metadata = {
                "timestamp": time.time() * 1000,
                "src_bytes": len(pkt[IP].payload),
                "dst_bytes": 0, # Placeholder for live sniffing direction
                "protocol_type": pkt[IP].proto,
                "raw_features": X_pca
            }
            
            self.callback(metadata)
        except Exception as e:
            # Silent failure for malformed packets
            pass
