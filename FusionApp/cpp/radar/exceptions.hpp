#pragma once

#include <exception>
#include <string>

namespace radar
{

/**
 * Base exception class for all radar-related errors
 */
class RadarException : public std::exception
{
  private:
    std::string _message;

  public:
    explicit RadarException(const std::string& message) : _message(message) {}

    const char* what() const noexcept override
    {
        return _message.c_str();
    }

    const std::string& getMessage() const noexcept
    {
        return _message;
    }
};

/**
 * @brief
 *
 */
class FileNotFoundException : public RadarException
{
  public:
    explicit FileNotFoundException(const std::string& filepath)
        : RadarException("File not found: " + filepath)
    {
    }
};

/**
 * Exception thrown when a frame is invalid or corrupted
 */
class InvalidFrameException : public RadarException
{
  public:
    explicit InvalidFrameException(const std::string& reason)
        : RadarException("Invalid frame: " + reason)
    {
    }
};

/**
 * Exception thrown when synchronization fails
 */
class SyncException : public RadarException
{
  public:
    explicit SyncException(const std::string& reason)
        : RadarException("Synchronization error: " + reason)
    {
    }
};

/**
 * Exception thrown when configuration is invalid
 */
class ConfigException : public RadarException
{
  public:
    explicit ConfigException(const std::string& reason)
        : RadarException("Configuration error: " + reason)
    {
    }
};

/**
 * Exception thrown when directory operations fail
 */
class DirectoryException : public RadarException
{
  public:
    explicit DirectoryException(const std::string& reason)
        : RadarException("Directory error: " + reason)
    {
    }
};

/**
 * Exception thrown when frame parsing fails
 */
class FrameParsingException : public RadarException
{
  public:
    explicit FrameParsingException(const std::string& reason)
        : RadarException("Frame parsing error: " + reason)
    {
    }
};

}  // namespace radar
