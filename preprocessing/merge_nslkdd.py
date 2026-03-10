import pandas as pd

# column names of NSL-KDD dataset
columns = [
"duration","protocol_type","service","flag","src_bytes",
"dst_bytes","land","wrong_fragment","urgent","hot",
"num_failed_logins","logged_in","num_compromised",
"root_shell","su_attempted","num_root","num_file_creations",
"num_shells","num_access_files","num_outbound_cmds",
"is_host_login","is_guest_login","count","srv_count",
"serror_rate","srv_serror_rate","rerror_rate",
"srv_rerror_rate","same_srv_rate","diff_srv_rate",
"srv_diff_host_rate","dst_host_count","dst_host_srv_count",
"dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
"dst_host_serror_rate","dst_host_srv_serror_rate",
"dst_host_rerror_rate","dst_host_srv_rerror_rate",
"label"
]

# load train dataset
train = pd.read_csv("data/raw/KDDTrain+.txt", names=columns)

# load test dataset
test = pd.read_csv("data/raw/KDDTest+.txt", names=columns)

# merge both
merged = pd.concat([train, test], ignore_index=True)

print("Merged dataset shape:", merged.shape)

# save merged dataset
merged.to_csv("data/processed/nslkdd_merged.csv", index=False)

print("Dataset saved in data/processed/")