class alpha_model:
    def __init__(self):
        self.signal=None
        self.alpha_data=None
    def ingest(self,data):
        self.alpha_data=data
    def generate_signal(self):
        if self.alpha_data is None:
            return None
        signal_df=self.alpha_data
        self.signal=signal_df[["Date","Ticker"]]
        self.signal["Signal"]=-signal_df["Short Interest % Float"]
    def run_model(self,current_data):
        self.ingest(current_data)
        self.generate_signal()