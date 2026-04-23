def run(auto_approve: bool = False) -> bool:
    if auto_approve:
        return True
    response = input("Approve script and continue? (y/n): ")
    return response.strip().lower().startswith("y")
