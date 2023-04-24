import sys
sys.path.append(".")
from sustaingym.envs.datacenter import DatacenterGym
from sustaingym.data.load_moer import MOERLoader
from datetime import datetime, timezone, timedelta


if __name__ == "__main__":
    # # 289 = 1 + 288 = 1 + (60min/5min)*24h
    # # Example MOER LOADER USAGE
    # startime = datetime(2019, 5, 1)
    # endtime = datetime(2019, 6, 1)
    # balancing_authority = "SGIP_CAISO_PGE"
    # save_dir = "sustaingym/data/moer/"
    # carbon_loader = MOERLoader(startime, endtime, balancing_authority, save_dir)
    # someday = datetime(2019, 5, 12, 0, 0, 0, 0, tzinfo=timezone(timedelta(0,0,0,0,0,0,0)))
    # data = carbon_loader.retrieve(someday)
    # print(data.shape)

    env = DatacenterGym()
    i = 0
    while True:
        h = i % 24
        d = min(h, 24-h)
        vcc = 0.8 - d*0.03
        obs, reward, terminal, truncated, info = env.step(vcc)
        if terminal:
            break
        print(f"STEP #{i}")
        env.render()
        print("")
        i += 1
