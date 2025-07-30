

class loggerCSV:
    def __init__(self, filename, mod):
        self.filename = filename
        self.file = open(self.filename, 'w')
        
        if mod == "cliff":
            self.file.write("Episode,Total Reward,Steps,Falls\n")
        elif mod == "cliff_mod":
            self.file.write("Episode,Total Reward,Steps,Falls,Rollbacks\n")
        elif mod == "taxi":
            self.file.write("Episode,TotalReward,Steps,drop_passsenger_pick_ghost,delivered_passenger\n")
        elif mod == "taxi_mod":
            self.file.write("Episode,TotalReward,Steps,drop_passsenger_pick_ghost,delivered_passenger,Rollbacks\n")
    
    def log_cliff(self, episode, total_reward, steps, falls):
        self.file.write(f"{episode},{total_reward},{steps},{falls}\n")
        self.file.flush()
    
    def log_cliff_Mod(self, episode, total_reward, steps, falls, rollbacks):
        self.file.write(f"{episode},{total_reward},{steps},{falls},{rollbacks}\n")
        self.file.flush()

    def log_taxi(self, episode, total_reward, steps, drop_passsenger_pick_ghost, delivered_passenger):
        self.file.write(f"{episode},{total_reward},{steps},{drop_passsenger_pick_ghost},{delivered_passenger}\n")
        self.file.flush()

    def log_taxi_Mod(self, episode, total_reward, steps, drop_passsenger_pick_ghost, delivered_passenger, rollbacks):
        self.file.write(f"{episode},{total_reward},{steps},{drop_passsenger_pick_ghost},{delivered_passenger},{rollbacks}\n")
        self.file.flush()
    

    def close(self):
        self.file.close()