Header header    # Header information
int64 req        # Request number
bool success     # True if the VLM successfully responded
string info      # Human-readable information (e.g., errors or status)
string response  # Text response from the VLM
string reasoning # Chain-of-Thought (CoT) reasoning or additional context
uint64 req_time  # Time prompt sent to VLM (nanoseconds)
uint64 res_time  # Time response received from VLM (nanoseconds)