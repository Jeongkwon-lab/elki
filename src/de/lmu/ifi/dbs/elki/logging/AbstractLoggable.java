package de.lmu.ifi.dbs.elki.logging;

import java.util.logging.LogRecord;

import de.lmu.ifi.dbs.elki.utilities.Progress;
import de.lmu.ifi.dbs.elki.utilities.Util;

/**
 * Abstract superclass for classes being loggable, i.e. classes intending to log
 * messages.
 * <p/>
 * 
 * @author Steffi Wanka
 */
public abstract class AbstractLoggable {
  static {
    LoggingConfiguration.assertConfigured();
  }

  /**
   * Holds the class specific debug status.
   */
  protected boolean debug;

  /**
   * The logger of the class.
   */
  protected final Logging logger;

  /**
   * Initializes the logger and sets the debug status to the given value.
   * 
   * @param debug the debug status.
   */
  protected AbstractLoggable(boolean debug) {
    this.logger = Logging.getLogger(this.getClass());
    this.debug = debug;
  }

  /**
   * Initializes the logger with the given name and sets the debug status to the
   * given value.
   * 
   * @param debug the debug status.
   * @param name the name of the logger.
   */
  protected AbstractLoggable(boolean debug, String name) {
    this.logger = Logging.getLogger(name);
    this.debug = debug;
  }

  /**
   * Log an exception at SEVERE level.
   * <p/>
   * If the logger is currently enabled for the SEVERE message level then the
   * given message is forwarded to all the registered output Handler objects.
   * 
   * Depreciated:
   * 
   * use {@link LoggingUtil.logExpensive(LogLevel.SEVERE, msg, e) instead.
   */
  public void exception(String msg, Throwable e) {
    logger.exception(msg, e);
  }

  /**
   * Log a WARNING message.
   * <p/>
   * If the logger is currently enabled for the WARNING message level then the
   * given message is forwarded to all the registered output Handler objects.
   * 
   * Depreciated:
   * 
   * use {@link LoggingUtil.logExpensive(LogLevel.WARNING, msg) instead.
   */
  public void warning(String msg) {
    logger.warning(msg);
  }

  /**
   * Log a PROGRESS message.
   * <p/>
   * If the logger is currently enabled for the PROGRESS message level then the
   * given message is forwarded to all the registered output Handler objects.
   */
  public void progress(Progress pgr) {
    logger.progress(pgr);
  }

  /**
   * Log a PROGRESS message.
   * <p/>
   * If the logger is currently enabled for the PROGRESS message level then the
   * given message is forwarded to all the registered output Handler objects.
   * 
   * @param pgr the progress to be logged
   * @param numClusters The current number of clusters
   * @see Loggable#progress(de.lmu.ifi.dbs.elki.utilities.Progress)
   */
  public void progress(Progress pgr, int numClusters) {
    logger.progress(new ProgressLogRecord(Util.status(pgr, numClusters), pgr.getTask(), pgr.status()));
  }

  /**
   * Log a PROGRESS message.
   * <p/>
   * If the logger is currently enabled for the PROGRESS message level then the
   * given message is forwarded to all the registered output Handler objects.
   */
  public void progress(LogRecord record) {
    logger.progress(record);
  }

  /**
   * Log a VERBOSE message.
   * <p/>
   * If the logger is currently enabled for the VERBOSE message level then the
   * given message is forwarded to all the registered output Handler objects.
   */
  public void verbose(String msg) {
    logger.verbose(msg);
  }

  /**
   * Log a DEBUG_FINE message.
   * <p/>
   * If the logger is currently enabled for the DEBUG_FINE message level then
   * the given message is forwarded to all the registered output Handler
   * objects.
   * 
   * Depreciated:
   * 
   * Use
   * 
   * <pre>
   * if (logger.isLoggable(LogLevel.FINE)) {
   *   logger.log(LogLevel.FINE, msg);
   * }
   * </pre>
   * 
   * instead. If msg is a constant, you can leave away the if statement.
   */
  public void debugFine(String msg) {
    logger.debugFine(msg);
  }

  /**
   * Log a DEBUG_FINER message.
   * <p/>
   * If the logger is currently enabled for the DEBUG_FINER message level then
   * the given message is forwarded to all the registered output Handler
   * objects.
   * 
   * Depreciated:
   * 
   * Use
   * 
   * <pre>
   * if (logger.isLoggable(LogLevel.FINER)) {
   *   logger.log(LogLevel.FINER, msg);
   * }
   * </pre>
   * 
   * instead. If msg is a constant, you can leave away the if statement.
   */
  public void debugFiner(String msg) {
    logger.debugFiner(msg);
  }

  /**
   * Log a DEBUG_FINEST message.
   * <p/>
   * If the logger is currently enabled for the DEBUG_FINEST message level then
   * the given message is forwarded to all the registered output Handler
   * objects.
   * 
   * Depreciated:
   * 
   * Use
   * 
   * <pre>
   * if (logger.isLoggable(LogLevel.FINEST)) {
   *   logger.log(LogLevel.FINEST, msg);
   * }
   * </pre>
   * 
   * instead. If msg is a constant, you can leave away the if statement.
   */
  public void debugFinest(String msg) {
    logger.debugFinest(msg);
  }
}
