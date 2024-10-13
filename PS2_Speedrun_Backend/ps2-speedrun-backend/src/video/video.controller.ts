import {
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
  Body,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { VideoService } from './video.service';
import { CreateVideoDto } from './dto/create-video.dto';
import { diskStorage } from 'multer';
import { extname } from 'path';
import { ApiConsumes, ApiBody, ApiTags, ApiOperation } from '@nestjs/swagger';
import { Express } from 'express';

@ApiTags('videos')
@Controller('videos')
export class VideoController {
  constructor(private readonly videoService: VideoService) {}

  @Post('upload')
  @UseInterceptors(
    FileInterceptor('video', {
      storage: diskStorage({
        destination: './uploads',
        filename: (req, file, cb) => {
          const filename = `${Date.now()}-${file.originalname}`;
          cb(null, filename);
        },
      }),
    }),
  )
  @ApiOperation({ summary: 'Upload a video file' })
  @ApiConsumes('multipart/form-data')
  @ApiBody({
    description: 'Video file upload',
    schema: {
      type: 'object',
      properties: {
        video: {
          type: 'string',
          format: 'binary',
        },
        userId: {
          type: 'string',
        },
      },
    },
  })
  async uploadFile(
    @UploadedFile() file: Express.Multer.File,
    @Body() createVideoDto: CreateVideoDto,
  ) {
    createVideoDto.videoUrl = `/uploads/${file.filename}`;
    createVideoDto.fileSize = file.size;
    return this.videoService.create(createVideoDto);
  }
}
