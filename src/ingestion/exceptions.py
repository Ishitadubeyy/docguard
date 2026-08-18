"""Exceptions raised during document ingestion."""


class DocumentIngestionError(Exception):
    """Base exception for document ingestion failures."""


class DocumentNotFoundError(DocumentIngestionError):
    """Raised when the requested document path does not exist."""


class UnsupportedDocumentTypeError(DocumentIngestionError):
    """Raised when the document format is not supported."""


class DocumentValidationError(DocumentIngestionError):
    """Raised when a document fails content validation."""
