# QnA 로그를 커맨드라인에서 검색·삭제할 수 있는 CLI 도구.
# `search` 서브커맨드로 로그를 조회하고, `purge` 서브커맨드로 보존 기간 초과 로그를 정리한다.
from __future__ import annotations

import argparse
import json
import sys

from services.qna_logging import purge_logs, search_logs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QnA log utility (search/purge)")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Search QnA logs")
    search.add_argument("--kind", choices=["query", "retrieval"], default="query")
    search.add_argument("--request-id", default=None)
    search.add_argument("--status", default=None)
    search.add_argument("--date", default=None, help="YYYY-MM-DD prefix")
    search.add_argument("--limit", type=int, default=20)

    purge = sub.add_parser("purge", help="Purge old logs by retention")
    purge.add_argument("--retention-days", type=int, default=90)
    purge.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview purge result without writing changes",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "search":
        rows = search_logs(
            kind=args.kind,
            request_id=args.request_id,
            status=args.status,
            date_prefix=args.date,
            limit=args.limit,
        )
        print(json.dumps({"count": len(rows), "items": rows}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "purge":
        result = purge_logs(retention_days=args.retention_days, dry_run=args.dry_run)
        print(json.dumps({"dry_run": args.dry_run, **result}, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
