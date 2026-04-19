FROM golang:1.24-bookworm AS builder

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY main.go ./

ENV CGO_ENABLED=0
RUN go build -o /listenai-stat .

FROM alpine:3.20

RUN apk add --no-cache ca-certificates

WORKDIR /app

COPY --from=builder /listenai-stat /app/listenai-stat

EXPOSE 8002

CMD ["/app/listenai-stat"]
