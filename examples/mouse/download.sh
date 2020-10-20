#!/bin/bash

# Scipt to batch download CRCNS.org data files stored at NERSC.
# Version 0.9, Oct 28, 2014

# Location of CRCNS.org files at NERSC
HOST_DIR="https://portal.nersc.gov/project/crcns/download"

# This script requires a separate file, called "crcns-account.txt"
# which contains lines like:
#crcns_username='<username>'
#crcns_password='<password>'
#(But without a '#' in front of the lines specifying the
#username and password.  Comment lines are prefixed by '#').

#####  Should not need to edit from here down

INFILE="filelist.txt"
ACCOUNT_FILE="crcns-account.txt"
HOST_SCRIPT=$HOST_DIR/index.php
COOKIE_FILE="cookie.txt"
COOKIE_NAME="crcns_nersc_download"
REGEX_ERROR="<p>(Error:[^<]*)<"

function login {
   # login if not loged in (if no cookie file) or if cookie is no longer working
   REGEX_SUCCESS="(Logged in as $USERNAME) "
   if [ -e "$COOKIE_FILE" ]; then
      echo "Previous $COOKIE_FILE file found.  Testing it."
      load_cookie
      url="$HOST_DIR/$DSID"
      cmd="curl -s -G -d $COOKIE_NAME=$sess_key $url"
      # cmd="curl -s -b $COOKIE_FILE -d fn=$DSID $HOST_SCRIPT"
      # echo "doing '$cmd'"
      # HTML=`curl -s -b $COOKIE_FILE -d fn=$DSID $HOST_SCRIPT`
      HTML=`$cmd`
      if [[ "$HTML" =~ $REGEX_ERROR ]]; then
         error_msg="${BASH_REMATCH[1]}";
         echo $error_msg;
         exit 1
      fi
      if [[ "$HTML" =~ $REGEX_SUCCESS ]]; then
         echo "Previous cookie worked."
         success_msg=${BASH_REMATCH[1]};
         echo $success_msg
         return
      fi
      echo "previous cookie did not work"
      # echo "returned html is"
      # echo $HTML
   fi
   echo "Trying to create new session."
   cmd="curl -s -c $COOKIE_FILE -d username=$USERNAME -d password=$PASSWORD -d fn=$DSID -d submit=Login $HOST_SCRIPT"
   # echo "doing '$cmd'"
   HTML=`$cmd`
   # HTML=`curl -s -c $COOKIE_FILE -d username=$USERNAME -d password=$PASSWORD -d fn=$DSID -d submit=Login $HOST_SCRIPT`
   if [[ "$HTML" =~ $REGEX_ERROR ]]; then
      error_msg=${BASH_REMATCH[1]};
      echo $error_msg;
      exit 1
   fi
   if [[ "$HTML" =~ $REGEX_SUCCESS ]]; then
      load_cookie
      success_msg=${BASH_REMATCH[1]};
      echo $success_msg
      return
   fi
   echo "Unable to create new session.  Returned HTML below."
   echo $HTML
   echo "Aborting"
   exit 1
}

function load_cookie {
   # load cookie so can use it as a URL parameter in a get request
   while read line
   do
      parts=($line)
      cookie_name=${parts[5]}
      sess_key=${parts[6]}
      if [[ "$cookie_name" = "$COOKIE_NAME" ]]; then
        continue
      fi
   done < $COOKIE_FILE
   if [[ ! "$cookie_name" = "$COOKIE_NAME" ]]; then
      echo "Unable to load cookie '$COOKIE_NAME' from file '$COOKIE_FILE'.  Aborting."
      exit 1
   fi
}

function load_account {
   # load account information from $ACCOUNT_FILE
   if [ ! -e "$ACCOUNT_FILE" ]; then
      echo "File '$ACCOUNT_FILE' not found.  Aborting."
      exit 1
   fi
   USERNAME=''
   PASSWORD=''
   RE_USERNAME="^crcns_username='(.*)'";
   RE_PASSWORD="^crcns_password='(.*)'";
   while read line
   do
      if [[ "$line" = "" ]] || [[ ${line:0:1} == "#" ]]; then
         continue
      fi
      if [[ "$line" =~ $RE_USERNAME ]]; then
         USERNAME="${BASH_REMATCH[1]}";
      elif [[ "$line" =~ $RE_PASSWORD ]]; then
         PASSWORD="${BASH_REMATCH[1]}";
      else
         echo "Unrecognize line in file '$ACCOUNT_FILE'.  Line is:"
         echo $line
         echo "Aborting."
         exit 1
      fi
   done < $ACCOUNT_FILE
   if [[ "$USERNAME" = "" ]] || [[ "$PASSWORD" = "" ]]; then
      echo "Unable to find non-empty username and password in file '$ACCOUNT_FILE'."
      echo "Suggestion: edit file '$ACCOUNT_FILE' to include your CRCNS.org account information."
      echo "Aborting."
      exit 1
   fi
}


function fetch {
   # download file given in $1
   FN="$DSID/$1"
   url="$HOST_DIR/$FN"
   AGENT=2  # indicates this batch download script
   cmd="curl -C - -G -d $COOKIE_NAME=$sess_key -d agent=$AGENT --write-out %{http_code} --create-dirs -o $FN $url"
   # echo "doing $cmd"
   status_code=`$cmd`
   # code 200 = regular download OK, 206 = partial download OK
   if [[ ! "$status_code" == "200" ]] && [[ ! "$status_code" = "206" ]]; then
      echo "** POSSIBLE ERROR, returned status code is $status_code"
      echo "Suggestion: Inspect file $FN to make sure it's correct."
      echo "Aborting."
      exit 1
   fi 
}

function mkSizeHuman {
   # convert size in bytes to human readable form
   x=$1
   if [[ "$x" -lt "1024" ]]; then
      return
   fi
   x=$[$x * 10]  # so can display tenths value
   s="BKMGTEP"
   while [[ "$x" -gt 10240 ]] && [[ ! "$s" = "" ]] ; do
     x=$[$x / 1024]
     s=${s:1}
   done
   intVal=$[$x / 10]
   decVal=$[$x - $intVal * 10]
   sizeHuman=" ($intVal.$decVal ${s:0:1}B)"
}


function load_files {
   # Parse file $INFILE (list of files with sizes) and either download or simulate download ($step=tally)
   # $step=tally to get count of files and bytes
   # $step=download to download files
   mode="default"
   download_count=0
   total_bytes=0
   header_displayed=""
   line_count=0
   while read line
   do
      line_count=$[$line_count + 1]
      # echo "processing line: $line"
      if [[ ${line:0:1} == "#" ]]; then
         # line starts with '#' - check for mode specify
         if [[ ${line:0:16} = "# mode='default'" ]]; then
            echo "found mode='default'"
            mode="default"
         elif [[ ${line:0:10} = "# mode='+'" ]]; then
            echo "found mode='+'"
            mode="+"
         fi
      else
         # line does not start with '#' - perhaps is file to download
         # display header if not already displayed
         if [[ "$header_displayed" = "" ]]; then
            if [ "$step" = "tally" ]; then
               echo "Would download files:"
            else
               echo "Downloading files:"
            fi
            echo -e "Name\tSize"
            header_displayed="1"
         fi
         # check for plus at start of line
         if [[ ${line:0:1} = "+" ]]; then
            foundPlus=1
            line=${line:1}
         else
            foundPlus=""
         fi
         # extract file name (fn), size (bytes), and human readable size (sizeh)
         parts=($line)
         fn=${parts[0]}
         size=${parts[1]}
         sizeh_1=${parts[2]}  # human readable size
         sizeh_2=${parts[3]}
         if [[ ! "$size_2" = "" ]]; then
            sizeh=" $sizeh_1 $sizeh_2"
            if [[ ! "$sizeh" =~ "^\([0-9]+ [KMGPTE]B\)$" ]]; then
               echo "** ERROR: File '$DSID/$INFILE' line $line_count is not in correct format."
               echo "Should contain file name and size, e.g.: ./fileName.txt 34, and optional human readable size"
               echo "but instead contains:"
               echo $line
               echo "Human readable size is not in correct format.  Aborting."
               echo "Suggestion: remove file '$DSID/$INFILE' then re-run this script to download a new version"
               exit 1
            fi
         else
            sizeh=""
         fi
         if [[ ! "$size" =~ ^[0-9]+ ]]; then
            echo "** ERROR: File '$DSID/$INFILE' line $line_count is not in correct format."
            echo "Should contain file name and size, e.g.: ./fileName.txt 34"
            echo "but instead contains:"
            echo $line
            echo "Suggestion: remove file '$DSID/$INFILE' then re-run this script to download a new version"
            echo "Aborting."
            exit 1
         fi
         if [ "$mode$foundPlus" = "+1" ] || [ "$mode" = "default" ]; then
            # is file to download, display name and size
            echo -en "$fn\t$size$sizeh"
            # see if already downloaded
            full_path="$DSID/$fn"
            # echo "full_path-$full_path"
            if [ -e "$full_path" ]; then
               fsize=`ls -ln $full_path | awk '{print $5}'`  # actual file size
               # echo "fsize=$fsize, size=$size, download_count=$download_count, total_bytes=$total_bytes "
               bytes_needed=`expr $size - $fsize`
               # echo "bytes_needed=$bytes_needed"
               if [ "$bytes_needed" = "0" ]; then
                  echo " - already downloaded.  Skipping"
                  continue
               elif [[ "$bytes_needed" -lt "0" ]]; then
                  echo " - saved file is $fsize bytes, which is larger than the file to download ($size) bytes"
                  echo "This should not happen.  Aborting."
                  exit 1
               else  
                  echo " - $fsize bytes downloaded; resuming download for $bytes_needed remaining."
               fi
            else
               bytes_needed=$size
               echo ""  # line return after file and size
            fi
            if [ "$step" = "download" ]; then
               fetch $fn
            fi
            download_count=$[$download_count +1]
            total_bytes=$[$total_bytes + $bytes_needed]
         fi
      fi
   done < $DSID/$INFILE
   mkSizeHuman $total_bytes
   if [ "$step" = "tally" ]; then
     echo "Will download $download_count files, $total_bytes bytes$sizeHuman"
     if [[ "$total_bytes" = 0 ]]; then
        echo "Nothing to download.  Exiting program."
        exit 0
     fi
   else
     echo "Done downloading $download_count files, $total_bytes bytes$sizeHuman"
   fi
}

function main {
   load_account
   login
   if [ ! -e "$DSID/$INFILE" ]; then
      echo "File $DSID/$INFILE not found.";
      echo "Fetching $DSID/$INFILE..."
      fetch "$INFILE"
      echo "$DSID/$INFILE was fetched."
      echo "edit $DSID/$INFILE to select files to download, then run this again to download the files"
   else
      step="tally"
      load_files
      read -p "Continue (enter 'y' if yes)?" yesno
      if [ ! "$yesno" == "y" ]; then
         echo "aborting."
         exit 0
      fi
      step="download"
      load_files
   fi
   echo "All done"
}

# start of script is here
if [ "$#" -ne 1 ]; then
    read -p "CRCNS.org dataset ID to fetch files? (example pvc-1): " DSID
else
    DSID="$1"
fi
main
exit 0




# if [ ! -n "$1" ]; then
#   echo "usage $0 <file_list_file>";
#  exit 0
# fi;

FILE_LIST=$1

if [ ! -e "$FILE_LIST" ]; then
  echo "File '$FILE_LIST' not found";
  exit 1;
fi

echo "all done"
exit 0




