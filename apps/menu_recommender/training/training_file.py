import random
import argparse
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta


class Preprocess:
    @staticmethod
    def make_data(number):
        modules = {
            "Sale": ["Opportunity", "Quotation", "Sale Order"],
            "Purchase": ["Purchase Request", "Purchase Order"],
            "Inventory": ["Goods Receipt", "Goods Detail", "Delivery"],
            "Report": ["Dashboard", "Report Inventory"],
        }

        # Luồng thực tế
        workflow = [
            "Opportunity",
            ("Quotation", 0.8),  # 80% user sẽ vào Quotation
            "Sale Order",
            ("Purchase Request", 0.8),  # 80% user sẽ vào Purchase Request
            "Purchase Order",
            "Goods Receipt",
            ("Goods Detail", 0.3),  # 30% user sẽ vào Delivery
            "Delivery",
            ("Dashboard", 0.65),  # 66% user vào Dashboard
            ("Report Inventory", 0.5)  # 50% user vào Report Inventory
        ]

        employee_ids = [f"user{i}" for i in range(1, 11)]
        view_types = ["list", "create", "detail", "update"]
        start_date = datetime(2024, 1, 1)

        data = []
        for _ in range(number):
            employee_id = random.choice(employee_ids)
            random_days = random.randint(0, 364)
            hour = random.randint(6, 20)
            minute = random.randint(0, 59)
            timestamp = start_date + timedelta(days=random_days, hours=hour, minutes=minute)

            # Tạo luồng thực tế
            session = []
            for step in workflow:
                if isinstance(step, tuple):  # Nếu có xác suất
                    if random.random() < step[1]:  # Kiểm tra xác suất
                        session.append(step[0])
                else:
                    session.append(step)

            # Sinh dữ liệu cho từng bước trong session
            for i in range(len(session)):
                function_name = session[i]
                module_name = next((key for key, val in modules.items() if function_name in val), "Unknown")
                view = random.choice(view_types)

                data.append({
                    "employee_id": employee_id,
                    "module": module_name,
                    "function_name": function_name,
                    "view_name": view,
                    "path": f"/{module_name.lower()}/{function_name.lower().replace(' ', '_')}/{view}",
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "time_slot": hour
                })

                # Tăng thời gian lên 5-15 phút mỗi khi chuyển sang chức năng tiếp theo
                timestamp += timedelta(minutes=random.randint(5, 15))

        # Lưu dữ liệu vào CSV
        df = pd.DataFrame(data)
        df.to_csv(r"D:\ML2AI\ML2AI_api\apps\menu_recommender\training\user_activity_log_realistic.csv", index=False)
        print(df.head(10))
        return True

    @staticmethod
    def markov_chains(df):
        transitions_matrix = {}  # Dictionary chứa ma trận của từng nhân viên
        for employee_id in df["employee_id"].unique():  # Lặp qua từng nhân viên
            employee_data = df[df["employee_id"] == employee_id]  # Lọc
            transitions_matrix[employee_id] = {}  # Tạo dictionary cho từng nhân viên
            for i in range(len(employee_data) - 1):
                current_state = employee_data.iloc[i]["function_name_with_time_slot"]
                next_state = employee_data.iloc[i + 1]["function_name_with_time_slot"]
                if current_state not in transitions_matrix[employee_id]:
                    transitions_matrix[employee_id][current_state] = {}
                if next_state not in transitions_matrix[employee_id][current_state]:
                    transitions_matrix[employee_id][current_state][next_state] = 0
                transitions_matrix[employee_id][current_state][next_state] += 1
        return transitions_matrix

    @staticmethod
    def markov_chains_normalize(transitions_matrix):
        normalized_matrix = {}  # Dictionary để lưu ma trận chuẩn hóa
        for employee_id, employee_matrix in transitions_matrix.items():  # Duyệt qua từng nhân viên
            normalized_matrix[employee_id] = {}  # Tạo dictionary mới cho nhân viên đó
            for state, transitions in employee_matrix.items():
                total = sum(transitions.values())  # Tổng số lần chuyển đổi từ trạng thái `state`
                normalized_matrix[employee_id][state] = {
                    next_state: count / total for next_state, count in transitions.items()
                }
        return normalized_matrix

    @staticmethod
    def process_and_save_to_mongo(reset_db=False):
        df = pd.read_csv(r'D:\ML2AI\ML2AI_api\apps\menu_recommender\training\user_activity_log_realistic.csv')
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["employee_id", "timestamp"])
        df['function_name_with_time_slot'] = df['function_name'].str.replace(' ', '_') + '#' + df['time_slot'].astype(str)

        transitions_matrix = Preprocess.markov_chains(df)
        transitions_matrix_normalize = Preprocess.markov_chains_normalize(transitions_matrix)

        client = MongoClient("mongodb://localhost:27017/")
        db = client["employee_data"]
        collection = db["transitions_matrix_normalize"]

        if reset_db:
            collection.delete_many({})
            print("Reset database!")

        documents = [
            {"employee_id": employee_id, "matrix": matrix}
            for employee_id, matrix in transitions_matrix_normalize.items()
        ]

        if documents:
            collection.insert_many(documents)
            print("Saved to MongoDB!")
            return True
        return False


class TrainModel:
    @staticmethod
    def train():
        return True


class Predict:
    @staticmethod
    def load_employee_matrix(employee_id):
        client = MongoClient("mongodb://localhost:27017/")
        db = client["employee_data"]
        collection = db["transitions_matrix_normalize"]
        data = collection.find_one({"employee_id": employee_id}, {"_id": 0, "matrix": 1})
        return data["matrix"] if data else None

    @staticmethod
    def predict(user_id: str, function_name: str, time_slot: int, top_k=2):
        emp_transitions_matrix_normalize = Predict.load_employee_matrix(user_id)
        if emp_transitions_matrix_normalize:
            function_name_with_time_slot = function_name.replace(' ', '_') + '#' + str(time_slot)
            if function_name_with_time_slot in emp_transitions_matrix_normalize:
                employee_matrix = emp_transitions_matrix_normalize[function_name_with_time_slot]
                top_functions = sorted(employee_matrix.items(), key=lambda x: x[1], reverse=True)[:top_k]
                return [func.split('#')[0].replace('_', ' ') for func, _ in top_functions]
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_data", type=bool, default=False, help="is make data")
    parser.add_argument("--number", type=int, default=10000, help="number of data created")
    parser.add_argument("--save_mongo", type=bool, default=False, help="process data and save to mongodb")
    parser.add_argument("--reset_db", type=bool, default=False, help="delete all db in mongodb")
    parser.add_argument("--predict", type=bool, default=True, help="is predict")
    parser.add_argument("--user_id", type=str, default='', help="user_id to predict")
    parser.add_argument("--function_name", type=str, default='', help="function_name to predict")
    parser.add_argument("--time_slot", type=str, default=0, help="time_slot to predict")
    args = parser.parse_args()

    if args.make_data and args.number:
        Preprocess.make_data(args.number)
    elif args.save_mongo and args.reset_db:
        Preprocess.process_and_save_to_mongo(args.reset_db)
    elif args.predict and args.user_id and args.function_name and args.time_slot:
        next_function = Predict.predict(args.user_id, args.function_name, args.time_slot)
        print(next_function)
    else:
        print('Error args format!')
