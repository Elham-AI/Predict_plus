# [API Name] API Documentation

## Getting Started

To get started with the [API Name], follow these steps:

**Base URL**: All API requests should be made to the base URL: `http://localhost:<port>`

## Endpoints

### predict

- **Description**: an endpoint to performe prediction.
- **URL**: `/predict`
- **Method**: `POST`
- **Auth required**: No

#### Input

- **Body Parameters**
```json
<input_json>
```

#### Output

- **Success Response**:
  - **Code**: `200 OK`
  - **Content**:
```json
{
  "prediction": ["<the prediction of first record>",
                 "<the prediction of second record>",
                 .....etc],
}
```

- **Error Response**:
  - **Code**: `400 BAD REQUEST`
  - **Content**:

```json
{
  "error": "Description of the error"
}
```
---
### ping

- **Description**: an endpoint for ping.
- **URL**: `/ping`
- **Method**: `GET`
- **Auth required**: No

#### Output

- **Success Response**:
  - **Code**: `200 OK`
  - **Content**:
```json
{
  "message":"healthy"
}
```