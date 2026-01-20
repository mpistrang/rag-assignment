# Example Page — Widget Manager

Auth Requirement: Authenticated
Feature Flags: WidgetFeature
Last Updated: January 1, 2026
Linked APIs: - GET /api/widgets

- POST /api/widgets
- DELETE /api/widgets/{id}
  Module: Examples
  Owner: Example Author
  Roles: Admin, Manager
  Route / URL: /app/examples/widgets
  Status: Example

### Purpose

This is an example documentation page for demonstration purposes only. It does not describe real functionality.

### Route

`/app/examples/widgets`

### Audience and Access

- Auth requirement: Authenticated
- Roles: Admin, Manager

### Primary Actions

- View list of example widgets
- Create a new example widget
- Delete an existing example widget

### Key UI States

- Empty state when no widgets exist
- Loading state while fetching data
- Error state when API fails

### Features & functionality

- Widget table with sorting and filtering
- Bulk selection for batch operations
- Export widgets to CSV format
- Import widgets from external sources
- Real-time updates via websocket connection

### Dependencies

- Widget API service
- Authentication service
- Notification service

### Related Pages

- Example Page — Widget Detail
- Example Page — Widget Settings

### Owner

@Example Author

### Last updated

January 1, 2026
