import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { VideoService } from './video.service';
import { VideoController } from './video.controller';
import { Video, VideoSchema } from './schema/video.schema';

@Module({
  imports: [
    MongooseModule.forFeature([{ name: Video.name, schema: VideoSchema }]), // Register Video schema
  ],
  controllers: [VideoController],
  providers: [VideoService],
})
export class VideoModule {}
