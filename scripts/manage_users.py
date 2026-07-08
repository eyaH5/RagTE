import asyncio
import argparse
import uuid
from api.database import async_session, User, Department, init_db
from api.auth import hash_password
from sqlalchemy import select, delete

async def list_users():
    async with async_session() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        print(f"{'Name':<20} {'Email':<30} {'Role':<10} {'Dept':<15} {'Active':<8}")
        print("-" * 85)
        for u in users:
            print(f"{u.name:<20} {u.email:<30} {u.role:<10} {u.department_id:<15} {str(u.is_active):<8}")

async def create_user(name, email, password, role, dept):
    async with async_session() as session:
        # Check if dept exists
        result = await session.execute(select(Department).where(Department.id == dept))
        if not result.scalar_one_or_none():
            print(f"Error: Department '{dept}' does not exist.")
            return

        new_user = User(
            id=str(uuid.uuid4()),
            email=email,
            name=name,
            password_hash=hash_password(password),
            department_id=dept,
            role=role,
            is_active=True
        )
        session.add(new_user)
        try:
            await session.commit()
            print(f"User {email} created successfully.")
        except Exception as e:
            print(f"Error: {e}")

async def delete_user(email):
    async with async_session() as session:
        result = await session.execute(delete(User).where(User.email == email))
        await session.commit()
        if result.rowcount > 0:
            print(f"User {email} deleted.")
        else:
            print(f"User {email} not found.")

async def update_password(email, new_password):
    async with async_session() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if user:
            user.password_hash = hash_password(new_password)
            await session.commit()
            print(f"Password updated for {email}.")
        else:
            print(f"User {email} not found.")

async def main():
    parser = argparse.ArgumentParser(description="Manage RAG users manually.")
    subparsers = parser.add_subparsers(dest="command")

    # List
    subparsers.add_parser("list", help="List all users")

    # Create
    create_parser = subparsers.add_parser("create", help="Create a new user")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--email", required=True)
    create_parser.add_argument("--password", required=True)
    create_parser.add_argument("--role", default="viewer", choices=["admin", "manager", "analyst", "viewer"])
    create_parser.add_argument("--dept", default="backoffice")

    # Delete
    delete_parser = subparsers.add_parser("delete", help="Delete a user")
    delete_parser.add_argument("--email", required=True)

    # Password
    pw_parser = subparsers.add_parser("password", help="Update user password")
    pw_parser.add_argument("--email", required=True)
    pw_parser.add_argument("--password", required=True)

    args = parser.parse_args()

    await init_db()

    if args.command == "list":
        await list_users()
    elif args.command == "create":
        await create_user(args.name, args.email, args.password, args.role, args.dept)
    elif args.command == "delete":
        await delete_user(args.email)
    elif args.command == "password":
        await update_password(args.email, args.password)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
